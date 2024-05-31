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
from merged_tokenizer import MergedTokenizer
from torch import nn
from augmentation import transpose_sm
import einops
from tqdm import tqdm
from util import top_k_top_p_filtering
from flow_utils import GaussianFourierProjection
import math

class TransposeWeightLayer(nn.Module):
    def __init__(self, tied_layer):
        super(TransposeWeightLayer, self).__init__()
        self.tied_layer = tied_layer

    def forward(self, x):
        return nn.functional.linear(x, self.tied_layer.weight.T)
    
class HierarchicalCausalDecoderModel(pl.LightningModule):
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
        same_encoder_decoder=False,
        tie_embedding_prior=False,
        full_mask_rate = 0.0,
        prior_embedding_bias=True,
        prior_logit_bias=False,
        sum_event_embedding_in_note_decoder=False,
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

        self.same_encoder_decoder = same_encoder_decoder
        if not self.same_encoder_decoder:
            self.prior_transformer = torch.nn.TransformerEncoder(
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

        self.event_transformer = torch.nn.TransformerEncoder(
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

        self.attribute_transformer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=feed_forward_size,
                norm_first=norm_first,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=1,
        )
        # make causal mask
            
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size)
        if tie_embedding_prior:
            self.prior_embedding_layer = TransposeWeightLayer(self.embedding_layer)
        else:
            self.prior_embedding_layer = torch.nn.Linear(vocab_size, hidden_size, bias=prior_embedding_bias)


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
        self.e_sos_z = torch.nn.Parameter(torch.randn(hidden_size))

        self.event_embedding = torch.nn.Parameter(torch.randn(self.n_events, hidden_size))
        
        self.format_mask = torch.Tensor(einops.rearrange(self.tokenizer.get_format_mask(), "(e a) v -> e a v", e=self.n_events))

        self.full_mask_rate = full_mask_rate

        self.prior_logit_bias = prior_logit_bias

        self.sum_event_embedding_in_note_decoder = sum_event_embedding_in_note_decoder

    
    def sample(self, mask, temperature=1.0, top_k=0, top_p=1.0, force_mask=True, reorder_mask=False):
        format_mask = einops.rearrange(self.format_mask, "e a v -> 1 e a v").to(self.device)
        x = torch.zeros(1, self.n_events, self.n_attributes, dtype=torch.long).to(self.device)
        mask = torch.Tensor(mask).to(self.device).float()
        mask = einops.rearrange(mask, "1 (e a) v -> 1 e a v", e=self.n_events)
        if reorder_mask:
            # sort mask by events sums
            event_mask_sums = mask.sum(-1).sum(-1)
            mask = mask[torch.arange(mask.shape[0]), event_mask_sums.argsort()]
            print(mask.shape)
  
        for i in tqdm(range(self.n_events)):
            event_z = self.event_forward(x[:,:-1], mask)
            for j in range(self.n_attributes):
                logits = self.attribute_forward(x[:,:,:-1], event_z)
                logits = logits[:,i,j]
                logits = logits / temperature
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                if force_mask:
                    probs += 1e-12
                    # mutliply with format mask
                    probs = probs * mask[:,i,j]
                    # normalize
                    probs = probs /probs.sum(-1, keepdim=True)
                else:
                    probs += 1e-12
                    probs = probs * format_mask[:,i,j]
                    probs = probs /probs.sum(-1, keepdim=True)
                x[:,i,j] = torch.multinomial(probs, 1).squeeze(-1)
        return x
    
    def event_forward(self, x, mask):
        '''
        Forward pass for the event layer.

        Args:
            x: (batch_size, events, attributes) tensor

        Returns:
            (batch_size, events, d) tensor
        '''
        # embed mask
        maskz = self.prior_embedding_layer(mask)
        maskz = maskz.sum(-2)
        # add event embedding
        ez = einops.rearrange(self.event_embedding.to(x.device), "e d -> 1 e d")
        maskz = maskz + ez

        xz = self.embedding_layer(x)
        xz = xz.sum(-2)
        b, t, d = xz.shape
        sos_z = einops.repeat(self.e_sos_z, "d -> b 1 d", b=b)
        xz = torch.cat([sos_z, xz], dim=1)
        xz = xz + ez
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            self.tokenizer.config["max_notes"]
        ).to(x.device)

        # for each layer, apply the transformers
        for i in range(self.event_transformer.num_layers):
            if not self.same_encoder_decoder:
                maskz = self.prior_transformer.layers[i](maskz)
            else:
                maskz = self.event_transformer.layers[i](maskz)
            xz = self.event_transformer.layers[i](xz, src_mask=causal_mask)
            # add maskz to xz
            xz = xz + maskz
        return xz

    def attribute_forward(self, x, event_z):
        '''
        Forward pass for the attribute_forward method.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, events, attributes-1).
            event_z (torch.Tensor): Input tensor of shape (batch_size, events, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, events, attributes, d).
        '''
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            self.n_attributes
        ).to(x.device)
        # embed
        xz = self.embedding_layer(x)
        if self.sum_event_embedding_in_note_decoder:
            xz = torch.cat([torch.zeros_like(event_z[...,None,:]), xz], dim=-2)
            xz += event_z[:,:,None,:]
        else:
            # concat event embedding
            xz = torch.cat([event_z[...,None,:], xz], dim=-2)
       
        # merge event into batch
        xz = einops.rearrange(xz, "b e a d -> (b e) a d")
        # through transformer
        xz_out = self.attribute_transformer(xz, mask=causal_mask, is_causal=True)
        # return logits
        logits = self.decoder_output_layer(xz_out)
        # reshape to (batch_size, events, attributes, d)
        logits = einops.rearrange(logits, "(b e) a d -> b e a d", b=x.size(0))
        return logits
    
    def forward(self, x, mask):
        event_z = self.event_forward(x[:,:-1], mask)
        logits = self.attribute_forward(x[:,:,:-1], event_z)
        # if mask == 0, set logits to -inf
        assert mask.shape == logits.shape
        if self.prior_logit_bias:
            logits[mask == 0] = -torch.inf
        return logits
    
    def step(self, batch, batch_idx):
        x = batch        
        targets = x
        inputs_ = einops.rearrange(x, "b (e a) -> b e a", e=self.n_events)

        b, e, a = inputs_.shape 

        x1h = torch.nn.functional.one_hot(inputs_, num_classes=len(self.vocab)).float()

        full_mask = torch.rand(b, device=self.device) < self.full_mask_rate
        full_mask = full_mask[:, None, None, None]

        masking_probs = torch.rand(b, device=self.device)
        position_mask = (
            torch.rand((b, e, a), device=self.device)
            < masking_probs[:, None,None]
        )

        # create masking ratios
        superposition_probs = torch.rand(b, device=self.device)
        superposition = torch.rand_like(x1h, device=self.device)<superposition_probs[:,None,None,None]

        mask = position_mask[:,:,:,None] * superposition

        masked_x = torch.clamp(x1h + mask + full_mask, 0, 1)

        # multiply by format mask
        masked_x = masked_x * self.format_mask[None, ...].to(masked_x.device)

        logits = self(inputs_, masked_x)
        logits = einops.rearrange(logits, "b e a d -> b (e a) d")
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
    
    def on_load_checkpoint(self, checkpoint):
        self.configure_optimizers()


if __name__ == "__main__":
    DATASET = "mmd_loops"

    BATCH_SIZE = 40

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


    # model = HierarchicalCausalDecoderModel(
    #     hidden_size=768,
    #     n_heads=8,
    #     feed_forward_size=2*768,
    #     n_layers=8,
    #     vocab=tokenizer.vocab,
    #     learning_rate=1e-4,
    #     tokenizer_config=tokenizer_config,
    #     learning_rate_gamma=0.99,
    #     norm_first=True,
    #     vocab_theta=False,
    #     warmup_steps=100,
    #     annealing_steps=100_000,
    #     min_lr_ratio=0.1,
    #     tied_embeddings=False,
    #     output_bias=False,
    #     use_adamw=False,
    #     same_encoder_decoder=True,
    #     full_mask_rate = 0.5,
    #     prior_embedding_bias=False,
    #     tie_embedding_prior=True,
    #     prior_logit_bias=False,
    #     sum_event_embedding_in_note_decoder=False,
    # )

    model = HierarchicalCausalDecoderModel.load_from_checkpoint(
        "./checkpoints/celestial-microwave-7/last.ckpt"
        ,
        map_location="cpu"
    )

    format_mask = torch.Tensor(tokenizer.get_format_mask())
    
    dummy_sample = format_mask.argmax(-1)

    model.step(dummy_sample[None, :].repeat(2, 1),0)

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

    wandb_logger = WandbLogger(log_model="all", project="causal-w-prior")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    progress_bar_callback = RichProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
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
        gradient_clip_val=1.0,
    )

    trainer.fit(
                model,
                trn_dl, 
                val_dl,
    )
                