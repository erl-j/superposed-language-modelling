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
import math

class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size,
        n_heads,
        feed_forward_size,
        n_layers,
        vocab,
        max_seq_len,
        learning_rate,
        tokenizer_config,
        one_hot_input=False,
        learning_rate_gamma=0.9,
        norm_first=False,
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
        # intialize positional encoding. one per step in sequence
        self.embedding_layer = nn.Linear(vocab_size, hidden_size)
        self.one_hot_input = one_hot_input
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=feed_forward_size,
            norm_first= norm_first,
            dropout=0.1,
            batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.decoder_output_layer = nn.Linear(hidden_size, vocab_size)
        
        self.seq_len = max_seq_len

        self.n_attributes = len(self.tokenizer.note_attribute_order)


        self.learning_rate_gamma = learning_rate_gamma

    def forward(self, x):
  
        format_mask = self.format_mask[None, ...].to(x.device)
        x = x * format_mask

        x = einops.rearrange(x, "b (t a) v -> b t a v", a=self.n_attributes)

        ze = self.embedding_layer(x)
        ze = ze.sum(dim=2)
        if self.use_positional_encoding:
            pos = self.positional_encoding[:, : self.tokenizer.config["max_notes"], :].to(self.device)
            if self.avg_positional_encoding:
                pos = pos.mean(dim=1, keepdim=True)
            ze = ze + pos
        # pass through transformer
        zl = self.transformer(ze)
        # get output part

        # note embeddings
        note_z = einops.rearrange(zl, "b t ft -> b t 1 ft")

        note_z = note_z.repeat(1, 1, self.n_attributes, 1)

        decoder_logits = self.decoder_output_layer(note_z)

        # multiply again by format mask
        decoder_logits[format_mask<0.5] = 1e-9
 
        # crop to decoder length
        return decoder_logits

    def schedule_fn(self, ratio):
        s = 1e-3
        f = lambda r : torch.cos(((r+s)/(1+s))*(torch.tensor(math.pi)/2))**2
        return f(ratio)/f(0)

    def step(self, batch, batch_idx):
        if self.one_hot_input:
            x = batch
        else:
            x = torch.nn.functional.one_hot(batch, num_classes=len(self.vocab)).float()

        target_idx = batch

        k=1

        batch_size, event_attr, v = x.shape

        # sample ratios uniformily for entire batch
        ratios = torch.rand(batch_size,device=x.device)

        # get schedule
        alphas = self.schedule_fn(ratios)

        s0 = x*(2*k)-k

        noise = torch.randn(x.shape)*k**2

        st = torch.sqrt(alphas)*s0 + torch.sqrt(1-alphas)*noise

        st_probs = torch.softmax(st,dim=-1)

        logits = self.model(st_probs)

        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_idx.reshape(-1),
        )

        metrics = {}
        metrics["cross_entropy"] = ce
        # TODO: check that this code is correct
        with torch.no_grad():
            # get probability of the correct token
            decoder_output_probs = F.softmax(logits, dim=-1)
            probability = torch.gather(
                decoder_output_probs, dim=-1, index=target_idx.unsqueeze(-1)
            ).squeeze(-1)
            metrics["probability"] = probability.mean()
            # sort yhat by probability
            decoder_output_probs_sort = torch.argsort(
                decoder_output_probs, dim=-1, descending=True
            )
            for k in [1, 2, 4]:
                metrics[f"accuracy@{k}"] = (
                    (
                        target_idx.unsqueeze(-1)
                        == decoder_output_probs_sort[:, :, :k]
                    )
                    .any(dim=-1)
                    .float()
                    .mean()
                )
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(f"trn/{metric}", metrics[metric], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        loss = metrics["cross_entropy"]
        self.log("trn/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        # log wandb name
        self.log("gpu", loss.device.index)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(f"val/{metric}", metrics[metric], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        loss = metrics["cross_entropy"]
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # learning rate decay
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.learning_rate_gamma, step_size=1)
        return [optimizer], [scheduler]


if __name__ == "__main__":

    genre_list = [
    "other",
    "pop",
    "rock",
    "italian%2cfrench%2cspanish",
    "classical",
    "romantic",
    "renaissance",
    "alternative-indie",
    "metal",
    "traditional",
    "country",
    "baroque",
    "punk",
    "modern",
    "jazz",
    "dance-eletric",
    "rnb-soul",
    "medley",
    "blues",
    "hip-hop-rap",
    "hits of the 2000s",
    "instrumental",
    "midi karaoke",
    "folk",
    "newage",
    "latino",
    "hits of the 1980s",
    "hits of 2011 2020",
    "musical%2cfilm%2ctv",
    "reggae-ska",
    "hits of the 1970s",
    "christian-gospel",
    "world",
    "early_20th_century",
    "hits of the 1990s",
    "grunge",
    "australian artists",
    "funk",
    "best of british"
    ]

    N_BARS = 4

    tokenizer_config = {
        "ticks_per_beat":24,
        "pitch_range":[0, 128],
        "max_beats":4*N_BARS,
        "max_notes":75 * N_BARS,
        "min_tempo":50,
        "max_tempo":200,
        "n_tempo_bins": 16,
        "n_velocity_bins": 32,
        "time_signatures": None,
        "tags": genre_list,
        "shuffle_notes": True,
        "use_offset": True,
        "merge_pitch_and_beat":False,
        "use_program": False,
        "use_instrument": True,
        "ignored_track_names":[f"Layers{i}" for i in range(0, 8)],
        "separate_drum_pitch": True,
        "use_drum_duration": False,
    }

    tokenizer = MergedTokenizer(
        tokenizer_config
    )

    model = DiffusionModel(
        hidden_size=768,
        n_heads=12,
        feed_forward_size=4 * 768,
        n_layers=12,
        vocab=tokenizer.vocab,
        max_seq_len=tokenizer.total_len,
        learning_rate=1e-4,
        tokenizer_config=tokenizer_config,
        learning_rate_gamma=0.99,
        norm_first=True,
    )


    # TODO : train on val for debugging only.
    trn_ds = MidiDataset(
        cache_path="./artefacts/val_midi_records_unique_pr.pt",
        path_filter_fn = lambda x: f"n_bars={N_BARS}" in x,
        genre_list=genre_list,
        tokenizer=tokenizer,
        transposition_range=[-4, 4],
        min_notes = 8*N_BARS,
        max_notes = tokenizer_config["max_notes"],
    )

    val_ds = MidiDataset(
        cache_path="./artefacts/val_midi_records_unique_pr.pt",
        path_filter_fn=lambda x: f"n_bars={N_BARS}" in x,
        genre_list=genre_list,
        tokenizer=tokenizer,
        min_notes=8 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
    )
    
    # desert capy uses batch size 80
    BATCH_SIZE = 80

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

    wandb_logger = WandbLogger(log_model="all", project="simplex-diffusion")
    # get name
    name = wandb_logger.experiment.name


    progress_bar_callback = RichProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
    accelerator="gpu",
    devices=[7],
    # precision="16-mixed",
    max_epochs=10_000,
    log_every_n_steps=1,
    # val_check_interval=10,
    callbacks=[
        # batch size finder
        progress_bar_callback,
            # learning rate monitor
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(
            dirpath=f"./checkpoints/{name}/",
            monitor="val/loss_epoch",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{step}-{val/loss_epoch:.5f}",
            )],
    logger=wandb_logger,
    gradient_clip_val=1.0,
    # accumulate_grad_batches=4,
    )

    trainer.fit(
        model,
        trn_dl,
        val_dl,
    )