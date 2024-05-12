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
from flow_utils import DirichletConditionalFlow, sample_cond_prob_path, expand_simplex, simplex_proj, GaussianFourierProjection
import time
from types import SimpleNamespace
import os

class SimpleFlow2Model(pl.LightningModule):
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
        # self.embedding_layer = nn.Linear(vocab_size, hidden_size, bias=False)
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
        self.E = nn.Parameter(torch.randn(vocab_size, hidden_size),requires_grad=True)
        self.U = nn.Parameter(torch.randn(hidden_size,vocab_size),requires_grad=True)
        # self.decoder_output_layer = nn.Linear(hidden_size, vocab_size, bias=output_bias)
        self.n_attributes = len(self.tokenizer.note_attribute_order)
        self.n_events = self.tokenizer.config["max_notes"]
        self.attribute_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(self.n_attributes)
        ])

        self.t_embedding = torch.nn.Linear(1, hidden_size)
        self.alphabet_size = len(vocab)

        self.iter_step = 0

        self.hidden_size = hidden_size

        self.time_embedder = nn.Sequential(nn.Linear(1, hidden_size), torch.nn.GELU(),nn.Linear(hidden_size, hidden_size))

    def z2p(self, z):
        p = torch.softmax(z@self.U,dim=-1)
        return p

    def p2z(self,p):
        z = p@self.E
        return z

    @torch.no_grad
    def sample(self, prior, n_steps, temperature):
        self.eval()
        B, L, V = 1, self.n_events * self.n_attributes, self.alphabet_size

        if prior is not None:
            prior = prior.to(self.device)
            z_prior = self.p2z(prior)

        zt = torch.randn(B, L, self.hidden_size, device=self.device)

        pts = []

        for step in tqdm(range(n_steps)):
            t = (step/n_steps) * torch.ones(B, device = self.device)
            delta_t = (1/n_steps) * torch.ones(B, device=self.device)


            pt = torch.softmax((zt@self.U),dim=-1)

            if prior is not None:
                pt = pt * prior
                pt = pt / (pt.sum(dim=-1, keepdim=True)+1e-12)
                # print min and max
                # assert torch.allclose(pt.sum(dim=-1),torch.ones(B,device=self.device))
                zt = (pt @ self.E)
            
                pt = torch.softmax((zt@self.U)/temperature,dim=-1)

            pts.append(pt.detach())

            v = self.forward(pt,t)

            zt = zt + delta_t * v
                
            logits = zt@self.U

            logits = torch.where(self.format_mask[None,...].to(self.device) < 0.5, -torch.inf, logits)

            # convert to probs
            p1 = torch.softmax(logits, dim=-1)


        pts = torch.stack(pts,dim=0)

        cmap_scale_fn = lambda x: x ** (1/4)

        print(pts.shape)

        plt.imshow(
            cmap_scale_fn(pts[:,0,:,:].mean(-2).cpu().T),
            cmap="magma", aspect="auto",interpolation="none")
        plt.title("pt")
        plt.show()

        # now plot pt diff
        pt_diff = pts[1:] - pts[:-1]
        plt.imshow(
            cmap_scale_fn(pt_diff[:,0,:,:].mean(-2).cpu().T),
            cmap="magma", aspect="auto",interpolation="none")
        plt.title("pt diff")
        plt.show()

        return p1.argmax(-1)
    
    def step(self, batch, batch_idx=None):
        x = batch
        x1h = torch.nn.functional.one_hot(batch, num_classes=self.alphabet_size).float()
        B, L, V = x1h.shape

        t = torch.rand(B, device=self.device)
        z1 = (x1h @ self.E)#.detach()
    
        z0 = torch.randn(B, L, self.hidden_size, device=self.device)

        zt = t[:,None,None] * z1 + (1-t[:,None,None]) * z0

        pt = self.z2p(zt)#.detach()

        v = self.forward(pt,t)


        mse_loss = ((v - (z1-z0))**2).mean()

        z1hat = zt + (1-t[:,None,None])*v

        logits = z1hat@self.U

        ce_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V),
            x.reshape(-1)
        ).mean()

        total_loss = mse_loss + ce_loss
        return  {
            "mse_loss":mse_loss,
            "ce_loss":ce_loss,
            "loss":total_loss,
        }
       
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
    
    def forward(self, s, t):
        s = s.clone()
        t = t.clone()
        t_z = self.time_embedder(t[:,None,None,None])
        format_mask = self.format_mask[None, ...].to(s.device)

        # double in last dimension
        x = s
        # sum across last dim
        x = format_mask * x
        # # normalize
        x = x / (x.sum(-1, keepdim=True)+1e-12)

        x = einops.rearrange(x, "b (t a) v -> b t a v", a=self.n_attributes)
        # detach here?
        ze = x @ self.E
        ze += t_z
        ze = ze.sum(dim=2)

        # pass through transformer
        zo = self.transformer(ze)
        # get output part
        # note embeddings
        za = einops.repeat(zo, "b n d-> b n a d", a=self.n_attributes)

        attr_z = []
        for i, layer in enumerate(self.attribute_layers):
            attr_z.append(layer(za[:, :, i, :]))
        za = torch.stack(attr_z, dim=2)
        # rearrange
        za = einops.rearrange(za, "b n a d -> b (n a) d")
        return za
    
    def configure_optimizers(self):
        # learning rate decay
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=self.hparams.learning_rate_gamma, step_size=1
        )
        return [optimizer], [scheduler]
    
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



    model = SimpleFlow2Model(
    hidden_size=512,
    n_heads=4,
    feed_forward_size=2*512,
    n_layers=6,
    vocab=tokenizer.vocab,
    tokenizer_config=tokenizer_config,
    one_hot_input=False,
    norm_first=True,
    activation = "gelu",
    output_bias = False,
    learning_rate=1e-3,
    learning_rate_gamma=0.98,
    )    

    # test step
    format_mask = torch.Tensor(tokenizer.get_format_mask())
    
    dummy_sample = format_mask.argmax(-1)

    model.step(dummy_sample[None, :].repeat(3, 1))

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

    wandb_logger = WandbLogger(log_model="all", project="simpleflow")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)
    wandb_logger.watch(model,log="all", log_freq=500)

    progress_bar_callback = RichProgressBar(refresh_rate=1)


    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[2],
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
        detect_anomaly=True,
        # accumulate_grad_batches=1,
    )

    #asd
    trainer.fit(
                model,
                trn_dl, 
                val_dl,
                # ckpt_path = "./checkpoints/dark-sky-67/last.ckpt"
    )