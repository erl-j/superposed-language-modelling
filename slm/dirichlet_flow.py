import datetime
import logging

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from data import MidiDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from merged_tokenizer import MergedTokenizer
from torch import nn
from augmentation import transpose_sm
import einops
from tqdm import tqdm
from util import top_k_top_p_filtering, top_p_probs
import numpy as np
from lightning.pytorch.utilities import grad_norm
import warnings
from train import EncoderOnlyModel
from flow_utils import DirichletConditionalFlow, sample_cond_prob_path, expand_simplex
import time
from types import SimpleNamespace

class DirichletFlowModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size,
        n_heads,
        feed_forward_size,
        n_layers,
        vocab,
        tokenizer_config,
        one_hot_input=False,
        norm_first=False,
        activation = "relu",
        output_bias = True,
        flow_args = {},
        learning_rate=1e-4,
        learning_rate_gamma=0.98,
    ):
        """
        seq_len: length of chart sequence (equal or longer to audio sequence)
        """
        super().__init__()
        self.save_hyperparameters()
        vocab_size = len(vocab)
        self.tokenizer = MergedTokenizer(tokenizer_config)
        self.format_mask = torch.Tensor(self.tokenizer.get_format_mask())
        self.vocab = vocab
        self.embedding_layer = nn.Linear(vocab_size, hidden_size, bias=False)
        self.one_hot_input = one_hot_input
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=feed_forward_size,
                norm_first=norm_first,
                dropout=0.1,
                batch_first=True,
                activation=activation
            ),
            num_layers=n_layers,
        )
        self.decoder_output_layer = nn.Linear(hidden_size, vocab_size, bias=output_bias)
        self.n_attributes = len(self.tokenizer.note_attribute_order)
        self.n_events = self.tokenizer.config["max_notes"]
        self.t_embedding = torch.nn.Linear(1, hidden_size)
        self.alphabet_size = len(vocab)

        self.flow_args = {
            "simplex_spacing":1000,
            "prior_pseudocount":2,
            "alpha_scale":2,
            "mode":"dirichlet",
            "alpha_max":8,
            "fix_alpha":None,
        }
        self.flow_args = SimpleNamespace(**self.flow_args)

        self.condflow = DirichletConditionalFlow(K=self.alphabet_size, alpha_spacing=0.01, alpha_max=self.flow_args.alpha_max)

        self.iter_step = 0
    def step(self, batch, batch_idx=None):
        self.iter_step += 1
        seq = batch
        seq_1hot = F.one_hot(seq, self.alphabet_size).float()
        B, L = seq.shape
        xt, alphas = sample_cond_prob_path(self.flow_args, seq, self.alphabet_size)
        logits = self.forward(xt, t=alphas)
        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), seq, reduction='none')
        losses = losses.mean(-1)
        self.last_log_time = time.time()
        return {"ce":losses.mean()}
    
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
        loss = metrics["ce"]
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
        loss = metrics["ce"]
        self.log(
            "val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss
    
    def forward(self, s, t):
        t_z = self.t_embedding(t[:,None,None])
        format_mask = self.format_mask[None, ...].to(s.device)
        x = s
        x = torch.nn.functional.softmax(x, dim=-1)
        x[format_mask.expand_as(x) < 0.5] = 0
        x = torch.nn.functional.softmax(x, dim=-1)

        x = einops.rearrange(x, "b (t a) v -> b t a v", a=self.n_attributes)
        ze = self.embedding_layer(x)
        ze += t_z[:,None,...]
        ze = ze.sum(dim=2)

        # pass through transformer
        zl = self.transformer(ze)
        # get output part
        # note embeddings
        note_z = einops.rearrange(zl, "b t ft -> b t 1 ft")
        note_z = note_z.repeat(1, 1, self.n_attributes, 1)
        decoder_logits = self.decoder_output_layer(note_z)
        # apply format mask
        decoder_logits = einops.rearrange(
            decoder_logits, "b t a v -> b (t a) v", a=self.n_attributes
        )
        decoder_logits[format_mask.expand_as(decoder_logits) < 0.5] = -1e12
        return decoder_logits
    
    def configure_optimizers(self):
        # learning rate decay
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=self.hparams.learning_rate_gamma, step_size=1
        )
        return [optimizer], [scheduler]
    
if __name__ == "__main__":

    DATASET = "clean_drums"

    BATCH_SIZE = 200

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

    model = DirichletFlowModel(
    hidden_size=128,
    n_heads=4,
    feed_forward_size=2*128,
    n_layers=6,
    vocab=tokenizer.vocab,
    tokenizer_config=tokenizer_config,
    one_hot_input=False,
    norm_first=False,
    activation = "relu",
    output_bias = False,
    learning_rate=1e-4,
    learning_rate_gamma=0.98,
    )    

    # test step
    format_mask = torch.Tensor(tokenizer.get_format_mask())

    print(format_mask.shape)
    
    dummy_sample = format_mask.argmax(-1)

    model.step(dummy_sample[None, :],0)

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

    wandb_logger = WandbLogger(log_model="all", project="symflow")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)
    wandb_logger.watch(model,log="all", log_freq=500)

    progress_bar_callback = RichProgressBar(refresh_rate=1)


    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[6],
        max_epochs=10_000,
        # log_every_n_steps=1,
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
                every_n_epochs=1 if DATASET == "mmd_loops" else 20,
            ),
        ],
        logger=wandb_logger,
        gradient_clip_val=1.0,
        # accumulate_grad_batches=1,
    )

    trainer.fit(
                model,
                trn_dl, 
                val_dl,
                # ckpt_path = "./checkpoints/dark-sky-67/last.ckpt"
    )