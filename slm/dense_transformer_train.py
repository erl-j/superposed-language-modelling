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
from dense_tokenizer import DenseTokenizer
from torch import nn
from augmentation import transpose_sm
from unet.unet import UNet2D
import einops

# model 


class DenseModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size,
        num_layers,
        n_heads,
        vocab,
        pitch_time_factorization,
        learning_rate,
        tokenizer_config,
        one_hot_input=False,
        beat_factorization=False,
    ):
        """
        seq_len: length of chart sequence (equal or longer to audio sequence)
        """
        super().__init__()
        self.save_hyperparameters()
        vocab_size = len(vocab)
        self.tokenizer = DenseTokenizer(tokenizer_config)
        self.format_mask = torch.tensor(self.tokenizer.get_format_mask())
        self.vocab = vocab

        self.beat_factorization = beat_factorization

        # intialize positional encoding. one per step in sequence
        self.voice_embedding = nn.Embedding(self.tokenizer.n_voices, hidden_size)

        self.n_beats = self.tokenizer.config["beats_per_bar"] * self.tokenizer.config["n_bars"]
        self.n_cells_per_beat = self.tokenizer.config["cells_per_beat"]

        if self.beat_factorization:
            self.beat_embedding = nn.Embedding(self.n_beats, hidden_size)
            self.cell_embedding = nn.Embedding(self.tokenizer.config["cells_per_beat"], hidden_size)

            self.beat_encoder = nn.TransformerEncoder(
                encoder_layer=torch.nn.TransformerEncoderLayer(
                    batch_first=True,
                    d_model=hidden_size,
                    dim_feedforward=hidden_size*4,
                    nhead=n_heads,
                    dropout=0.1,
                ),
                num_layers=1,
                norm = nn.LayerNorm(hidden_size),
            )
            self.beat_decoder = nn.TransformerEncoder(
                encoder_layer=torch.nn.TransformerEncoderLayer(
                    batch_first=True,
                    d_model=hidden_size,
                    dim_feedforward=hidden_size*4,
                    nhead=n_heads,
                    dropout=0.1,
                ),
                num_layers=1,
                norm = nn.LayerNorm(hidden_size),
            )

        else:
            self.time_embedding = nn.Embedding(self.tokenizer.timesteps, hidden_size)

        self.embedding_layer = nn.Linear(vocab_size, hidden_size, bias=False)
        self.one_hot_input = one_hot_input

        self.pitch_time_factorization = pitch_time_factorization

        self.main_block = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                batch_first=True,
                d_model=hidden_size,
                dim_feedforward=hidden_size*4,
                nhead=n_heads,
                dropout=0.1,
            ),
            num_layers=num_layers,
            norm = nn.LayerNorm(hidden_size),
        )

        self.decoder_output_layer = nn.Linear(hidden_size, vocab_size)

  
    
    def forward(self, x):


        # multiply with format mask
        x = x * self.format_mask[None, ...].to(x.device)

        constraint = x

        # embed
        x = self.embedding_layer(x)

        batch,time,voice,ch,ft = x.shape
        
        # sum channels
        x = x.sum(dim=-2)
        # add voice and time embeddings
        x += self.voice_embedding.weight[None, None, :, :].to(x.device)

        if not self.beat_factorization:
            x += self.time_embedding.weight[None, :, None, :].to(x.device)

        if self.beat_factorization:
            x = einops.rearrange(x, "batch (beat cell) voice ft -> (batch beat voice) cell ft", voice=voice, ft=ft, beat=self.n_beats, cell=self.tokenizer.config["cells_per_beat"])
            # add cell embedding
            x += self.cell_embedding.weight[None, :, :].to(x.device)
            x = self.beat_encoder(x).mean(dim=1)
            x = einops.rearrange(x, "(batch beat voice) ft -> batch beat voice ft", voice=voice, ft=ft, beat=self.n_beats)
            # add beat embedding
            x += self.beat_embedding.weight[None, :, None, :].to(x.device)

        if self.pitch_time_factorization or self.pitch_time_factorization =="pitch_time":  
            for layer in range(len(self.main_block.layers)):
                x_voice = einops.rearrange(x, "batch time voice ft -> (batch time) voice ft",  voice=voice, ft=ft, batch=batch)
                x_voice = self.main_block.layers[layer](x_voice)
                x_voice = einops.rearrange(x_voice, "(batch time) voice ft -> batch time voice ft",  voice=voice, ft=ft, batch=batch)

                x_time = einops.rearrange(x, "batch time voice ft -> (batch voice) time ft",  voice=voice, ft=ft, batch=batch)
                x_time = self.main_block.layers[layer](x_time)
                x_time = einops.rearrange(x_time, "(batch voice) time ft -> batch time voice ft",  voice=voice, ft=ft, batch=batch)

                # sum
                x = x_voice + x_time

            if self.main_block.norm is not None:
                x = self.main_block.norm(x)

        else:
            x = einops.rearrange(x, "batch time voice ft -> batch (time voice) ft", time=time, voice=voice, ft=ft)
            x = self.main_block(x)
            x = einops.rearrange(x, "batch (time voice) ft -> batch time voice ft", time=time, voice=voice, ft=ft)

        if self.beat_factorization:
            x = einops.rearrange(x, "batch beat voice ft -> (batch beat voice) 1 ft", voice=voice, ft=ft, beat=self.n_beats)
            # repeat in cell dimension
            x = x.repeat(1, self.tokenizer.config["cells_per_beat"], 1)
            # add cell embedding
            x += self.cell_embedding.weight[None, :, :].to(x.device)
            x = self.beat_decoder(x)
            x = einops.rearrange(x, "(batch beat voice) cell ft -> batch (beat cell) voice ft", voice=voice, ft=ft, beat=self.n_beats)
        # repeat x to match ch
        x = x[..., None, :].repeat(1, 1, 1, ch, 1)

        # output layer
        logits = self.decoder_output_layer(x)

        # multiply by constraint
        logits = logits - (1- constraint) * 1e5
       
        return logits
    
    def step(self, batch, batch_idx):
        batch = batch.long()
        if self.one_hot_input:
            x = batch
        else:
            x = torch.nn.functional.one_hot(batch, num_classes=len(self.vocab)).float()

        batch_size = x.shape[0]
        # create masking ratios
        masking_ratios = torch.rand(batch_size, device=self.device)
        mask = torch.rand_like(x, device=self.device)<masking_ratios[:,None,None,None,None]
        # mask encoder input
        # encoder input is mask or token tensor with undefined
        encoder_input = torch.clamp(x + mask, 0, 1)

        decoder_output_logits = self(encoder_input)

        ce = F.cross_entropy(
            decoder_output_logits.reshape(-1, decoder_output_logits.shape[-1]),
            batch.reshape(-1),
        )



        # multiply by batch
        # multiply by mask

        metrics = {}
        metrics["cross_entropy"] = ce
        
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(f"trn/{metric}", metrics[metric], prog_bar=True)
        loss = metrics["cross_entropy"]
        self.log("trn/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(f"val/{metric}", metrics[metric], prog_bar=True, on_step=True, on_epoch=True)
        loss = metrics["cross_entropy"]
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    

    def generate(self, x, sampling_steps, temperature):
        x = x
        # multiply by format mask
        x = x * self.format_mask[None, ...].to(x.device)
        batch, time, voice, ch, ft = x.shape
        # linspace from schedule
        schedule = torch.linspace(0, 1, sampling_steps, dtype=torch.float32, device=x.device)**2
        total_tokens = time * voice * ch 
        # count number of known tokens
        masked_tokens = (x.sum(-1) > 1).sum().int()
        # find masking ratio
        masking_ratio = masked_tokens / total_tokens

        # find step in schedule
        step = torch.argmin((1- schedule - masking_ratio).abs())

        print(f"step: {step}")
        for i in range(step+1, sampling_steps):

            print(f"step: {i}")

            logits = self(x.float())

            probs = F.softmax(logits/temperature, dim=-1)
            # invert probs
            # flatten
            flat_probs = einops.rearrange(probs, "b t v c l -> (b t v c) l")

            sampled = torch.multinomial(flat_probs, 1).squeeze(-1)

            print(f"sampled: {sampled.shape}")


            flat_x = einops.rearrange(x, "b t v c l -> (b t v c) l")

            masked_indices = torch.where(flat_x.sum(-1) > 1)[0]
            # 
            n_masked = masked_indices.shape[0]
            print(f"n_masked: {n_masked}")
            target_masking_ratio = 1-schedule[i].item()

            target_n_masked = int(total_tokens * target_masking_ratio)
            # tokens to unmask

            n_tokens_to_unmask = n_masked - target_n_masked

            print(f"n_tokens_to_unmask: {n_tokens_to_unmask}")

            # get indices of tokens to unmask
            # get indices of masked tokens
            # shuffle masked indices
            masked_indices = masked_indices[torch.randperm(n_masked)]
            indices_to_unmask = masked_indices[:n_tokens_to_unmask]


            # replace with sampled values
            flat_x[indices_to_unmask] = torch.nn.functional.one_hot(sampled[indices_to_unmask], num_classes=flat_x.shape[-1])

            x = einops.rearrange(flat_x, "(b t v c) l -> b t v c l", b=batch, t=time, v=voice, c=ch)
        return x





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
        "beats_per_bar": 4,
        "cells_per_beat": 12,
        "pitch_range": [0, 128],
        "n_bars": N_BARS,
        "max_notes": 100 * N_BARS,
        "min_tempo": 50,
        "max_tempo": 200,
        "n_tempo_bins": 16,
        "time_signatures": None,
        "tags": ["pop"],
        "shuffle_notes": True,
        "use_offset": True,
        "merge_pitch_and_beat": True,
        "use_program": True,
        "ignored_track_names": [f"Layers{i}" for i in range(0, 8)],
    }

    tokenizer = DenseTokenizer(
        tokenizer_config
    )

    trn_ds = MidiDataset(
        cache_path="./artefacts/trn_midi_records.pt",
        path_filter_fn = lambda x: f"n_bars={N_BARS}" in x,
        genre_list=genre_list,
        tokenizer=tokenizer,
        transposition_range=[-4, 4],
    )

    val_ds = MidiDataset(
        cache_path="./artefacts/val_midi_records.pt",
        path_filter_fn = lambda x: f"n_bars={N_BARS}" in x,
        genre_list=genre_list,
        tokenizer=tokenizer,
    )
  
    BATCH_SIZE = 2

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
    
    
    model = DenseModel(
        hidden_size=512,
        num_layers=8,
        pitch_time_factorization=True,
        n_heads=8,
        vocab = tokenizer.vocab,
        learning_rate=1e-4,
        tokenizer_config=tokenizer_config,
        beat_factorization=True
    )

    wandb_logger = WandbLogger(log_model="all", project="slm-dense")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    progress_bar_callback = RichProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
    accelerator="gpu",
    devices=[3],
    precision=32,
    max_epochs=None,
    log_every_n_steps=1,
    # val_check_interval=10,
    callbacks=[
            progress_bar_callback,
            pl.callbacks.ModelCheckpoint(
            dirpath=f"./checkpoints/{name}/",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{step}-{val/loss:.2f}-{trn/loss:.2f}",
            train_time_interval = datetime.timedelta(minutes=30),),
            # checkpoint based on train loss
            pl.callbacks.ModelCheckpoint(
            dirpath=f"./checkpoints/{name}/",
            monitor="trn/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{step}-{trn/loss:.2f}",
            train_time_interval = datetime.timedelta(minutes=5),),
            ],
    logger=wandb_logger,
    accumulate_grad_batches=4,
    )

    trainer.fit(model,
     trn_dl,
     val_dl,
    )