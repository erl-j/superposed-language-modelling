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
from util import top_k_top_p_filtering, top_p_probs
import numpy as np
from lightning.pytorch.utilities import grad_norm
import warnings
from train import EncoderOnlyModel

# def _cleanup_handler():
#     for f in _cleanups:
#         f()

# import atexit as _atexit
# _atexit.register(_cleanup_handler)

# warnings.filterwarnings("ignore", category=ResourceWarning)

class SimplexDiffusionModel(pl.LightningModule):
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
        normalize_by_masking_ratio=False,
        learning_rate_gamma=0.9,
        norm_first=False,
        k=5.0,
        beta_schedule="linear",
        activation = "relu",
        output_bias = True,
        format_mask_on_probs=False,
        relative_loss = False,
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
        self.learning_rate_gamma = learning_rate_gamma

        self.t_embedding = torch.nn.Linear(1, hidden_size)

        self.k = k

        self.beta_schedule = beta_schedule

        self.format_mask_on_probs = format_mask_on_probs

        self.relative_loss = relative_loss

    def forward(self, s, t, mask=None, prior_strength = 1.0):
        t_z = self.t_embedding(t)
        format_mask = self.format_mask[None, ...].to(s.device)
        x = s
        if self.format_mask_on_probs:
            if mask is not None:

                mode = "b"

                # A
                if mode == "a":
                    x = torch.nn.functional.softmax(x, dim=-1)
                    # set format mask to zeros
                    x[format_mask.expand_as(x) < 0.5] = 0
                    # renormalize
                    x = x / x.sum(dim=-1, keepdim=True)
                    x = (1-prior_strength)*x + prior_strength*mask
                    x = x / x.sum(dim=-1, keepdim=True)
                elif mode == "b":
                    # x = torch.nn.functional.softmax(x, dim=-1)

                    x = torch.nn.functional.softmax(x, dim=-1)
                    # set format mask to zeros
                    x[format_mask.expand_as(x) < 0.5] = 0
                    # renormalize
                    x = x / x.sum(dim=-1, keepdim=True)
                    uniform_prior = format_mask.expand_as(x).float()/format_mask.sum(dim=-1,keepdim=True).float()
                    prior = (1-prior_strength)*uniform_prior + prior_strength*mask
                    x = x * prior
                    x = x / x.sum(dim=-1, keepdim=True)

                # plt.plot(x[0,0].cpu().numpy())
                # plt.show()
                # asd
            else:
                x = torch.nn.functional.softmax(x, dim=-1)
                # set format mask to zeros
                x[format_mask.expand_as(x) < 0.5] = 0
                # renormalize
                x = x / x.sum(dim=-1, keepdim=True)

        else:
            x[format_mask.expand_as(x) < 0.5] = -self.k
            # softmax mask
            x = torch.nn.functional.softmax(x, dim=-1)

      

        x = einops.rearrange(x, "b (t a) v -> b t a v", a=self.n_attributes)
        ze = self.embedding_layer(x)
        ze += t_z[..., None, :]
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
        if self.format_mask_on_probs:
            decoder_logits[format_mask.expand_as(decoder_logits) < 0.5] = -1e12
        else:
            decoder_logits[format_mask.expand_as(decoder_logits) < 0.5] = -self.k
        return decoder_logits

    def discrete_output_distribution(self, theta, t):
        output_dist = self.forward(theta, t)
        output_probs = torch.nn.functional.softmax(output_dist, dim=-1)
        return output_probs

    def test_step(self, batch_size):
        format_mask = self.format_mask[None, ...].to(self.device)
        noise = torch.randn(
            (batch_size, self.n_events * self.n_attributes, len(self.vocab)),
            device=self.device,
        )
        noise = noise * format_mask.expand_as(noise)
        batch = noise.argmax(dim=-1)
        loss = self.step(batch, 0)
        return loss
    
    def schedule(self, t):
        if self.beta_schedule == "linear":
            return (1-t)
        elif self.beta_schedule == "cosine":
            s = 1e-4
            f_t = lambda tx : torch.cos(((tx+s)/(1+s))*(np.pi/2))**2
            return f_t(t)/f_t(torch.zeros_like(t))
        
    def plot_ce_curve(self, batch, batch_idx, tmp_k):
        if self.one_hot_input:
            x = batch
        else:
            target = batch
            x = torch.nn.functional.one_hot(batch, num_classes=len(self.vocab)).float()

        BATCH_SIZE = 300
        t = torch.linspace(0,1,BATCH_SIZE).unsqueeze(1).unsqueeze(1).to(self.device)
        alpha = self.schedule(t)

        k = tmp_k if tmp_k is not None else self.k

        s = (x * 2 - 1) * k

        batch_size, note_attr, vocab_size = s.shape

        noise = torch.randn((BATCH_SIZE, note_attr,vocab_size), device=s.device)*k

        s_n =  torch.sqrt(alpha) * s + torch.sqrt(1 - alpha) * noise

        # softmax
        p_n = torch.nn.functional.softmax(s_n, dim=-1)

        # get cross entropy
        cross_entropy = -(x * torch.log(p_n)).sum(dim=-1).sum(dim=-1)
        entropy = -(p_n*torch.log(p_n)).sum(dim=-1).sum(dim=-1)

        # cross entropy
        
        plt.plot(cross_entropy.cpu().numpy())
        plt.ylim(0,torch.max(cross_entropy).cpu().numpy())
        plt.title(f"Cross Entropy for k={k}")
        plt.show()

        plt.plot(entropy.cpu().numpy())
        plt.ylim(0,torch.max(entropy).cpu().numpy())
        plt.title(f"Entropy for k={k}")
        plt.show()

        for aidx,a in enumerate(self.tokenizer.note_attribute_order):
            plt.plot(p_n[:,aidx,:].mean(dim=0).cpu().numpy())
            plt.title(f"Attribute {a} for k={k}")
            plt.show()

        for aidx,a in enumerate(self.tokenizer.note_attribute_order):
            ce = -(x[:,aidx::self.n_attributes,:] * torch.log(p_n[:,aidx::self.n_attributes,:])).sum(dim=-1).sum(dim=-1)
            plt.plot(ce.detach().cpu().numpy())
            plt.title(f"Entropy & attribute {a} for k={k}")
            plt.show()

        note_probs = p_n[:,:self.n_attributes]

        note_probs = torch.log(note_probs)

        # create subplots
        fig, axs = plt.subplots(self.n_attributes,1, figsize=(4,2*self.n_attributes))

        for aidx,a in enumerate(self.tokenizer.note_attribute_order):
            axs[aidx].imshow(note_probs[:,aidx].cpu().numpy().T, aspect="auto",interpolation="none")
            # put subplot title
            axs[aidx].set_title(f"Attribute {a} for k={k}")
        plt.show()

    def step(self, batch, batch_idx):
        if self.one_hot_input:
            x = batch
        else:
            target = batch
            x = torch.nn.functional.one_hot(batch, num_classes=len(self.vocab)).float()

        s = (x * 2 - 1)* self.k

        batch_size, note_attr, vocab_size = s.shape

        t = torch.rand((batch_size, 1, 1), device=s.device)

        alpha = self.schedule(t)

        noise = torch.randn(s.shape, device=s.device)*self.k
        s_n =  torch.sqrt(alpha) * s + torch.sqrt(1 - alpha) * noise

        # forward pass
        s_hat = self.forward(s_n, t)

        if not self.relative_loss:
            cross_entropy = torch.nn.functional.cross_entropy(
                s_hat.reshape(-1,vocab_size),
                target.flatten(),
            )

        if self.relative_loss:
            base_cross_entropy = torch.nn.functional.cross_entropy(
                    s_n.reshape(-1,vocab_size),
                    target.flatten(),
                    reduction = 'none'
            )

            cross_entropy = torch.nn.functional.cross_entropy(
                s_hat.reshape(-1,vocab_size),
                target.flatten(),
                reduction = 'none'
            )

            base_cross_entropy = einops.rearrange(base_cross_entropy,"(b s)-> b s", b=batch_size)
            cross_entropy = einops.rearrange(cross_entropy,"(b s)-> b s", b=batch_size)

            base_cross_entropy = base_cross_entropy.mean(dim=-1)
            cross_entropy = cross_entropy.mean(dim=-1)

            cross_entropy  = cross_entropy / (base_cross_entropy+1e-5)

            cross_entropy = cross_entropy.mean()

        metrics = {"ce":cross_entropy}
      
        return metrics

    def self_eval(
            self,
            x,
            prior,
            prior_strength,
            t=1,
            decay_prior=False,
    ):
        
        batch_size, na = x.shape

        t = torch.ones((batch_size,1,1),device=self.device) * t

        x1h = torch.nn.functional.one_hot(x, num_classes=len(self.vocab)).float()

        s = (x1h * 2 - 1)* self.k

        vocab_size = len(self.vocab)

        alpha_t = self.schedule(t)

        t = torch.ones((batch_size,1,1),device=self.device) * 0.5

        alpha = self.schedule(t)

        noise = torch.randn(s.shape, device=s.device)*self.k
        s_n =  torch.sqrt(alpha) * s + torch.sqrt(1 - alpha) * noise

        wl = self.forward(noise,t, prior, prior_strength*(1-alpha_t) if decay_prior else prior_strength)

        # compute cross entropy
        cross_entropy = torch.nn.functional.cross_entropy(
            wl.reshape(-1,vocab_size),
            x.flatten(),
            reduction='none'
        ).reshape(batch_size,na).mean(dim=-1)

        return cross_entropy

    @torch.no_grad()
    def sample2(
        self,
        prior,
        batch_size,
        nb_steps,
        top_p,
        prior_strength,
        enforce_prior,
        plot=False,
        decay_prior=False
    ):
        self.eval()
        with torch.no_grad():
            # repeat to batch_size
            if prior is not None:
                format_mask = self.format_mask[None, ...].to(self.device)
                prior = prior*format_mask
                # normalize
                prior = prior / prior.sum(dim=-1, keepdim=True)
                if prior.shape[0] != batch_size:
                    prior = prior.repeat(batch_size,1,1)
                prior = prior.to(self.device).float()
                prior_flat = einops.rearrange(prior, "b s v -> (b s) v")
                prior_simplex = (prior * 2 - 1)*self.k


            t = torch.ones((batch_size,1,1),device=self.device)
            alpha = self.schedule(t)
            noise = torch.randn((batch_size, self.n_events*self.n_attributes,len(self.vocab)),device=self.device) * self.k
                
            wt = noise
            w0ps = []

            alphas = []

            for step in tqdm(range(nb_steps,-1,-1)):

                t = torch.ones((batch_size,1,1),device=self.device) * (step/nb_steps)

                alpha_t = self.schedule(t)

                wl = self.forward(wt,t, prior, prior_strength*(1-alpha_t) if decay_prior else prior_strength)

                wl = einops.rearrange(wl, "b s v -> (b s) v", s = self.n_events*self.n_attributes)                

                if plot:
                    preview_probs = torch.nn.functional.softmax(wl, dim=-1)
                    w0ps.append(einops.rearrange(preview_probs, "(b s) v -> b s v", s = self.n_events*self.n_attributes).cpu())
                
                # sample
                w0p = torch.nn.functional.softmax(wl, dim=-1)

                if enforce_prior and prior is not None:
                    w0p = (w0p+1e-9) * prior_flat
                    # if very low prop in prior then 0
                    # w0p = (w0p * (prior_flat>1e-9).float()) +1e-9
                    w0p = w0p / w0p.sum(dim=-1, keepdim=True)


                w0p = top_p_probs(w0p, top_p)


                # sample
                w0 = torch.multinomial(w0p, 1).squeeze(-1)

                w0x = einops.rearrange(w0, "(b s) -> b s", s = self.n_events*self.n_attributes)

                # convert back to simplex
                w0 = torch.nn.functional.one_hot(w0x, num_classes=len(self.vocab)).float()

                w0 = (w0 * 2 - 1) * self.k

                noise = torch.randn((batch_size, self.n_events*self.n_attributes,len(self.vocab)),device=self.device) * self.k
                
                t_minus_1 = torch.ones((batch_size,1,1),device=self.device) * ((step-1)/nb_steps)
                alpha = self.schedule(t_minus_1)
                alphas.append(alpha[0].item())
                wt = torch.sqrt(alpha) * w0 + torch.sqrt(1 - alpha) * noise

             
            #     sts.append(st.detach())
            #     pts.append(pt.detach())
            
            # sts = torch.stack(sts)
            # pts = torch.stack(pts)

            if plot:
                print(len(w0ps))
                wops = torch.stack(w0ps)

                # for aidx,a in enumerate(self.tokenizer.note_attribute_order):
                #     attribute_token_idxs = [i for i,v in enumerate(self.tokenizer.vocab) if a+":" in v]
                #     attribute_tokens = [self.tokenizer.vocab[i] for i in attribute_token_idxs]
                #     # set attribute token indices on y axis
                #     plt.imshow(wops[:,0,aidx,attribute_token_idxs].cpu().numpy().T, aspect="auto", cmap="rocket",interpolation="none")
                #     plt.yticks(range(len(attribute_tokens)),attribute_tokens)
                #     plt.title(f"Attribute {a}")
                #     plt.show()

                for aidx,a in enumerate(self.tokenizer.note_attribute_order):
                    attribute_token_idxs = [i for i,v in enumerate(self.tokenizer.vocab) if a+":" in v]
                    attribute_tokens = [self.tokenizer.vocab[i] for i in attribute_token_idxs]
                    # set attribute token indices on y axis
                    plt.imshow(wops[:,0,aidx::self.n_attributes,attribute_token_idxs].mean(-2).cpu().numpy().T, aspect="auto", cmap="rocket",interpolation="none")
                    plt.yticks(range(len(attribute_tokens)),attribute_tokens)
                    plt.title(f"Attribute {a}, all notes")
                    plt.show()


                entropy = -torch.sum(wops*torch.log(wops+1e-9),dim=-1).mean(dim=1).mean(dim=1)

                # log scale
                plt.plot(np.log(entropy.cpu().numpy()))
                plt.title("Entropy")
                plt.show()


            plt.plot(alphas)
            plt.title("Alpha")
            plt.show()

            # sample categorical
            return w0x

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
    
    def transplant_weights(self, other_model):
        self.embedding_layer.weight.data = other_model.embedding_layer.weight.data
        self.decoder_output_layer.weight.data = other_model.decoder_output_layer.weight.data
        if self.decoder_output_layer.bias is not None:
            self.decoder_output_layer.bias.data = other_model.decoder_output_layer.bias.data
        self.transformer.load_state_dict(other_model.transformer.state_dict())

    def configure_optimizers(self):
        # learning rate decay
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=self.learning_rate_gamma, step_size=1
        )
        return [optimizer], [scheduler]

if __name__ == "__main__":

    BATCH_SIZE = 20

    tag_list = open("./data/mmd_loops/tags.txt").read().splitlines()

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

    model = SimplexDiffusionModel(
        hidden_size=768,
        n_heads=12,
        feed_forward_size=4 * 768,
        n_layers=12,
        vocab=tokenizer.vocab,
        max_seq_len=tokenizer.total_len,
        learning_rate=1e-3,
        tokenizer_config=tokenizer_config,
        learning_rate_gamma=0.99,
        norm_first=True,
        k=5.0,
        beta_schedule="cosine",
        activation="gelu",
        output_bias=False,
        format_mask_on_probs=True
    )

    # other_model = EncoderOnlyModel.load_from_checkpoint(
    #     checkpoint_path = "./checkpoints/magic-forest-321/last.ckpt",
    #     device="cpu"
    # )

    # model.transplant_weights(
    #     other_model
    # )

    # model = SimplexDiffusionModel.load_from_checkpoint(
    #             checkpoint_path = "./checkpoints/serene-sunset-44/last.ckpt",
    #             learning_rate = 1e-4,
    #             map_location="cpu"
    # )
    # 80
    # model.test_step(batch_size=60)

    trn_ds = MidiDataset(
        cache_path="./data/mmd_loops/trn_midi_records_unique_pr.pt",
        path_filter_fn=lambda x: f"n_bars={N_BARS}" in x,
        genre_list=tag_list,
        tokenizer=tokenizer,
        transposition_range=[-4, 4],
        min_notes=8 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
    )

    val_ds = MidiDataset(
        cache_path="./data/mmd_loops/val_midi_records_unique_pr.pt",
        path_filter_fn=lambda x: f"n_bars={N_BARS}" in x,
        genre_list=tag_list,
        tokenizer=tokenizer,
        min_notes=8 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
    )

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

    wandb_logger = WandbLogger(log_model="all", project="simplex-diffusion")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)
    wandb_logger.watch(model,log="all", log_freq=500)

    progress_bar_callback = RichProgressBar(refresh_rate=1)


    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[7],
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
        # accumulate_grad_batches=1,
    )

    trainer.fit(
                model,
                trn_dl, 
                val_dl,
                ckpt_path="./checkpoints/flowing-paper-64/last.ckpt"
            
                # ckpt_path = "./checkpoints/fanciful-planet-7/last.ckpt"
    )
                # ckpt_path="checkpoints/upbeat-dawn-53/epoch=14-step=24330-val/loss_epoch=0.00273.ckpt")