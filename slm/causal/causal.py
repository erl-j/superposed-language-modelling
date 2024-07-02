import datetime
import logging

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from data import MidiDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from slm.data.merged_tokenizer import MergedTokenizer
from torch import nn
from slm.data.augmentation import transpose_sm
import einops
from tqdm import tqdm
from util import top_k_top_p_filtering
from slm.dirichlet_flow.flow_utils import GaussianFourierProjection
import math

class CausalDecoderModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size,
        n_heads,
        feed_forward_size,
        n_layers,
        vocab,
        learning_rate,
        tokenizer_config,  
        warmup_steps=1_000,
        annealing_steps=200_000,
        min_lr_ratio=0.1,
        one_hot_input=False,
        tied_embeddings=False,
        learning_rate_gamma=0.9,
        norm_first=False,
        vocab_theta=True,
        fourier_t_embedding=False,
        output_bias=True,
        use_adamw=True,
    ):
        """
        seq_len: length of chart sequence (equal or longer to audio sequence)
        """
        super().__init__()
        self.save_hyperparameters()
        vocab_size = len(vocab)
        self.tokenizer = MergedTokenizer(tokenizer_config)
        self.vocab = vocab
        self.one_hot_input = one_hot_input
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=feed_forward_size,
                norm_first=norm_first,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=n_layers,
        )

        # make causal mask
        
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.decoder_output_layer = nn.Linear(hidden_size, vocab_size, bias=output_bias)

        self.n_attributes = len(self.tokenizer.note_attribute_order)
        self.n_events = self.tokenizer.config["max_notes"]
        self.vocab_theta = vocab_theta
        self.warmup_steps = warmup_steps
        self.annealing_steps = annealing_steps
        self.min_lr_ratio = min_lr_ratio
        self.learning_rate = learning_rate
        self.learning_rate_gamma = learning_rate_gamma
        # add a sos token
        self.sos_z = torch.nn.Parameter(torch.randn(hidden_size))

        self.event_embedding = torch.nn.Parameter(torch.randn(self.n_events, hidden_size))
        self.attribute_embedding = torch.nn.Parameter(torch.randn(self.n_attributes, hidden_size))
        
    def forward(self, x):
        # remove last token
        xz = self.embedding_layer(x)

        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            self.tokenizer.config["max_notes"]
        )
        b, t, d = xz.shape
        sos_z = einops.repeat(self.sos_z, "d -> b 1 d", b=b)
        # concat with sos token
        xz = torch.cat([sos_z, xz], dim=1)
        # add event and attribute embeddings
        ez = einops.repeat(self.event_embedding, "e d -> 1 (e a) d", a=self.n_attributes)
        az = einops.repeat(self.attribute_embedding, "a d -> 1 (e a) d", e=self.n_events)
        xz = xz + ez + az        
        xz_out = self.transformer(xz, mask=causal_mask, is_causal=True)
        logits = self.decoder_output_layer(xz_out)
        return logits
    
    def step(self, batch, batch_idx):
        x = batch
        inputs = x[:, :-1]
        targets = x
        logits = self(inputs)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.flatten()
        )
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(
                f"trn/{metric}",
                metrics[metric],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        loss = metrics["loss"]
        self.log(
            "trn/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        # log wandb name
        self.log("gpu", loss.device.index)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(
                f"val/{metric}",
                metrics[metric],
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        loss = metrics["loss"]
        self.log(
            "val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss
    
    def configure_optimizers(self):
        if self.hparams.use_adamw:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        # get number of batches per epoch

        def warmup_cosine_lambda(current_step):
            # Assumes 1 epoch = len(train_dataloader) steps
            num_warmup_steps = self.warmup_steps
            num_annealing_steps = self.annealing_steps
            min_lr_ratio = self.min_lr_ratio

            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step < num_warmup_steps + num_annealing_steps:
                progress = (current_step - num_warmup_steps) / num_annealing_steps
                return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            else:
                return min_lr_ratio

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda step: warmup_cosine_lambda(
                step
        ))

        return [ optimizer ], [{"scheduler": scheduler,"interval":"step"}]


if __name__ == "__main__":
    DATASET = "mmd_loops"

    BATCH_SIZE = 100

    tag_list = open(f"./data/{DATASET}/tags.txt").read().splitlines()

    N_BARS = 4

    tokenizer_config = {
        "ticks_per_beat": 24,
        "pitch_range": [0, 128],
        "max_beats": 4 * N_BARS,
        "max_notes": 75 * N_BARS,
        "min_tempo": 50,
        "max_tempo": 200,
        "n_tempo_bins": 16,
        "n_velocity_bins": 32,
        "time_signatures": None,
        "tags": tag_list,
        "shuffle_notes": True,
        "use_offset": True,
        "merge_pitch_and_beat": False,
        "use_program": False,
        "use_instrument": True,
        "ignored_track_names": [f"Layers{i}" for i in range(0, 8)],
        "separate_drum_pitch": True,
        "use_drum_duration": False,
    }

    tokenizer = MergedTokenizer(tokenizer_config)


    model = CausalDecoderModel(
        hidden_size=32,
        n_heads=1,
        feed_forward_size=512,
        n_layers=24,
        vocab=tokenizer.vocab,
        learning_rate=1e-2,
        tokenizer_config=tokenizer_config,
        learning_rate_gamma=0.99,
        norm_first=True,
        vocab_theta=False,
        warmup_steps=1_000,
        annealing_steps=200_000,
        min_lr_ratio=0.1,
        tied_embeddings=False,
        output_bias=False,
        use_adamw=False,
    )

    # format_mask = torch.Tensor(tokenizer.get_format_mask())
    
    # dummy_sample = format_mask.argmax(-1)

    # model.step(dummy_sample[None, :].repeat(BATCH_SIZE, 1),0)

    mmd_4bar_filter_fn = lambda x: f"n_bars={N_BARS}" in x

    trn_ds = MidiDataset(
        cache_path=f"./data/{DATASET}/trn_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn if DATASET == "mmd_loops" else None,
        genre_list=tag_list,
        tokenizer=tokenizer,
        transposition_range=[-4, 4] if DATASET == "mmd_loops" or DATASET == "harmonic" else None,
        min_notes=8 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
    )
    # print len of dataset
    print(f"Loaded {len(trn_ds)} training records")

    val_ds = MidiDataset(
        cache_path=f"./data/{DATASET}/val_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn if DATASET == "mmd_loops" else None,
        genre_list=tag_list,
        tokenizer=tokenizer,
        min_notes=8 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
    )
    print(f"Loaded {len(val_ds)} validation records")

    # desert capy uses batch size 80
    trn_dl = torch.utils.data.DataLoader(
        trn_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    wandb_logger = WandbLogger(log_model="all", project="causal")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    progress_bar_callback = RichProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[4],
        max_epochs=10_000,
        log_every_n_steps=1,
        callbacks=[
            progress_bar_callback,
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(
                dirpath=f"./checkpoints/{name}/",
                monitor="val/loss_epoch",
                mode="min",
                save_top_k=3,
                save_last=True,
                filename="{epoch}-{step}-{val/loss_epoch:.5f}",
            ),
        ],
        logger=wandb_logger,
        # gradient_clip_val=1.0,
    )

    trainer.fit(
                model,
                trn_dl, 
                val_dl,
    )
                