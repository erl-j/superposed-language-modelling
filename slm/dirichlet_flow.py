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

class DirichletFlowModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size,
        n_heads,
        feed_forward_size,
        n_layers,
        vocab,
        tokenizer_config,
        flow_args,
        one_hot_input=False,
        norm_first=False,
        activation = "relu",
        output_bias = True,
        learning_rate=1e-4,
        learning_rate_gamma=0.98,
        fourier_t_embedding=False,
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

        self.flow_args = SimpleNamespace(**flow_args)
        self.iter_step = 0

        self.fourier_t_embedding = fourier_t_embedding
        if fourier_t_embedding:
            self.time_embedder = nn.Sequential(GaussianFourierProjection(embedding_dim= hidden_size),nn.Linear(hidden_size, hidden_size))

    @torch.no_grad()
    def generate(self, prior, generation_args):
        prior = prior.to(self.device)
        # assert that prior is normalized
        assert torch.allclose(prior.sum(-1), torch.ones_like(prior.sum(-1)))
        B = prior.shape[0]
        L = self.n_events * self.n_attributes
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, self.alphabet_size, device=self.device)).sample()
        eye = torch.eye(self.alphabet_size).to(x0)
        xt = x0

        self.condflow = DirichletConditionalFlow(K=self.alphabet_size, alpha_spacing=generation_args.alpha_spacing, alpha_max=generation_args.alpha_max)

        t_span = torch.linspace(1, generation_args.alpha_max, generation_args.num_integration_steps, device=self.device)

        xts = []

        xts.append(xt.detach().clone())

        out_probss = []

        for i, (s, t) in tqdm(enumerate(zip(t_span[:-1], t_span[1:]))):

            if not self.fourier_t_embedding:
                alphas = s
                ts = (alphas-1)/(generation_args.alpha_max-1)
            else:
                ts = s

            logits = self.forward(xt, ts[None].expand(B))

            # print(f'xt.min(): {xt.min()} xt.max(): {xt.max()} logits.min(): {logits.min()} logits.max(): {logits.max()}')
            out_probs = torch.nn.functional.softmax(logits / generation_args.flow_temp, -1)

            out_probs = out_probs * prior
            out_probs = out_probs / out_probs.sum(-1, keepdim=True)

            out_probss.append(out_probs.detach().clone())

            # add some noise
            # out_probs = out_probs + torch.randn_like(out_probs) * 0.001           

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {xt.min()}, out_probs.min(): {out_probs.min()}')
                c_factor = torch.nan_to_num(c_factor)
                out_logits = logits
                out_probs = out_probs
                break

            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (out_probs.unsqueeze(-2) * cond_flows).sum(-1)

            print(f'flow.min(): {flow.min()} flow.max(): {flow.max()}')

            xt = xt + flow * (t - s)
            # norma
            if not torch.allclose(xt.sum(2), torch.ones((B, L), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative.')
                xt = simplex_proj(xt)
                out_logits = logits
                out_probs = out_probs
                break

            xts.append(xt.detach().clone())

        xts = torch.stack(xts, 0)

        cmap_scale_fn = lambda x : x ** (1/4)

        # plot xts diff in first dimension
        xts_diff = xts[1:] - xts[:-1]

        plt.imshow(
            xts_diff[:,0,:,:].mean(-2).cpu().numpy().T,
            cmap='magma', aspect='auto', interpolation="none")
        plt.show()

        plt.imshow(
            cmap_scale_fn(xts[:,0,1,:].cpu().numpy().T),
            cmap='magma', aspect='auto', interpolation="none")
        plt.show()

        out_probss = torch.stack(out_probss, 0)

        plt.imshow(
            cmap_scale_fn(out_probss[:,0,1,:].cpu().numpy().T), cmap='magma', aspect='auto', interpolation="none")
        plt.show()


        plt.imshow(
            cmap_scale_fn(out_probss[:,0,1::self.n_attributes,:].mean(dim=-2).cpu().numpy().T), 
            cmap='magma', aspect='auto', interpolation="None")
        plt.show()

        # plot t_span with a line at the last i with a label
        plt.plot(t_span.cpu().numpy())
        plt.axvline(i, color='r', label=f"i_f={i}")
        plt.legend()
        plt.show()

        return out_logits, out_probs
    
    def step(self, batch, batch_idx=None):
        seq = batch
        seq_1hot = F.one_hot(seq, self.alphabet_size).float()
        B, L = seq.shape
        xt, alphas = sample_cond_prob_path(self.flow_args, seq, self.alphabet_size)
        if self.iter_step == 0:

            os.makedirs("artefacts/dflow", exist_ok=True)
            # plot xt[0]
            plt.imshow(xt[0].cpu().numpy())
            plt.savefig(f"artefacts/dflow/xt_{self.iter_step}.png")
            plt.close()

            plt.plot(xt[0,0].cpu().numpy())
            plt.savefig(f"artefacts/dflow/xt0_{self.iter_step}.png")
            plt.close()

            # plot alphas[0]
            plt.plot(alphas.cpu().numpy())
            plt.savefig(f"artefacts/dflow/alphas_{self.iter_step}.png")
            plt.close()

            # plot sorted alphas
            alphas_s, idx = alphas.sort()
            plt.plot(alphas_s.cpu().numpy())
            plt.savefig(f"artefacts/dflow/alphas_sorted_{self.iter_step}.png")
            plt.close()

            # plot entropy
            entropy = (-xt * torch.log(xt)).sum(-1).mean(-1)
            # sort
            entropy, idx = entropy.sort()
            plt.plot(entropy.cpu().numpy())
            plt.savefig(f"artefacts/dflow/entropy_{self.iter_step}.png")
            plt.close()

            # plot cross entropy
            ce = (-seq_1hot * torch.log(xt)).sum(-1).mean(-1)
            # sort
            ce, idx = ce.sort()
            plt.plot(ce.cpu().numpy())
            plt.savefig(f"artefacts/dflow/ce_{self.iter_step}.png")
            plt.close()

            # plot uniform cross entropy
            uniform = torch.ones_like(xt) / self.alphabet_size

            uce = (-seq_1hot * torch.log(uniform)).sum(-1).mean(-1)
            # sort
            uce, idx = uce.sort()
            plt.plot(uce.cpu().numpy())
            plt.savefig(f"artefacts/dflow/uce_{self.iter_step}.png")
            plt.close()

            # get xt with largest alpha
            xt_max_alpha = xt[alphas.argmax()]
            plt.plot(xt_max_alpha[0].cpu().numpy())
            plt.savefig(f"artefacts/dflow/xt_max_alpha_{self.iter_step}.png")
            plt.close()

            # get xt with smallest alpha
            xt_min_alpha = xt[alphas.argmin()]
            plt.plot(xt_min_alpha[0].cpu().numpy())
            plt.savefig(f"artefacts/dflow/xt_min_alpha_{self.iter_step}.png")
            plt.close()

            # get xt of median alpha
            # find median index
            alphas_s, idx = alphas.sort()
            median_idx = len(alphas) // 2
            plt.plot(xt[idx[median_idx]][0].cpu().numpy())
            plt.savefig(f"artefacts/dflow/xt_median_alpha_{self.iter_step}.png")
            plt.close()

        # xt, prior_weights = expand_simplex(xt, alphas, self.flow_args.prior_pseudocount)
        if not self.fourier_t_embedding:
            ts = (alphas-1)/(self.flow_args.alpha_max-1)
        else:
            ts = alphas
        logits = self.forward(xt, t=ts)
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.alphabet_size),
            seq.reshape(-1))
        self.last_log_time = time.time()
        self.iter_step += 1
        return {"ce":losses}
    
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
        s = s.clone()
        t = t.clone()

        if self.fourier_t_embedding:
            t_z = self.time_embedder(t)
        else:
            t_z = self.t_embedding(t[:,None])
        format_mask = self.format_mask[None, ...].to(s.device)


        # double in last dimension
        x = s
        # sum across last dim
        x[format_mask.expand_as(x) < 0.5] = 0
        # normalize
        x = x / (x.sum(-1, keepdim=True)+1e-12)

        x = einops.rearrange(x, "b (t a) v -> b t a v", a=self.n_attributes)
        ze = self.embedding_layer(x)
        ze += t_z[:, None, None, :]
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

    DATASET = "mmd_loops"

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

    flow_args = {
            "prior_pseudocount":2,
            # sampled during training
            "alpha_scale":10.0,
            "mode":"dirichlet",
            "fix_alpha":None,
    }


    model = DirichletFlowModel(
    hidden_size=512,
    n_heads=4,
    feed_forward_size=2*512,
    n_layers=6,
    vocab=tokenizer.vocab,
    flow_args=flow_args,
    tokenizer_config=tokenizer_config,
    one_hot_input=False,
    norm_first=True,
    activation = "gelu",
    output_bias = False,
    learning_rate=1e-3,
    learning_rate_gamma=0.98,
    fourier_t_embedding=True,
    )    

    # test step
    format_mask = torch.Tensor(tokenizer.get_format_mask())
    
    dummy_sample = format_mask.argmax(-1)

    model.step(dummy_sample[None, :].repeat(BATCH_SIZE, 1))

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
        devices=[3],
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

    #asd
    trainer.fit(
                model,
                trn_dl, 
                val_dl,
                # ckpt_path = "./checkpoints/dark-sky-67/last.ckpt"
    )