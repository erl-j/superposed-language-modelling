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


class BFNModel(pl.LightningModule):
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
        beta1=1.0,
        vocab_theta=True,
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
            ),
            num_layers=n_layers,
        )
        self.decoder_output_layer = nn.Linear(hidden_size, vocab_size)
        self.n_attributes = len(self.tokenizer.note_attribute_order)
        self.n_events = self.tokenizer.config["max_notes"]
        self.learning_rate_gamma = learning_rate_gamma

        self.t_embedding = torch.nn.Linear(1, hidden_size)

        self.beta1 = beta1

        self.vocab_theta = vocab_theta

    def forward(self, theta, t):
        t_z = self.t_embedding(t)
        format_mask = self.format_mask[None, ...].to(theta.device)
        theta = theta * format_mask
        # renormalize
        # theta = theta / theta.sum(dim=-1, keepdim=True)
        x = theta

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
        decoder_logits[format_mask.expand_as(decoder_logits) <0.5] = -1e9
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
    
    def preview_beta(self,batch):
        betas = [0.05,0.1,0.2,0.5]
        T = 10

        # linspace from 0 to 1 and includes 0 and 1
        t = torch.linspace(0,1,T) 
        assert t[0] == 0 # append 0
        assert t[-1] == 1

        # append 1
        

        BETAS = len(betas)

        # make subplot for each t, each beta
        fig, axs = plt.subplots(T,BETAS,figsize=(12,8))
        
        entropies = torch.zeros((T,BETAS))
        for beta_idx, beta in enumerate(betas):
            for i in range(T):
                theta = self.step(batch,0, tmp_beta1=beta,tmp_t=t[i], preview_input_dist=True)
                # calculate entropy
                entropies[i,beta_idx] = -torch.sum(theta * torch.log(theta + 1e-12),dim=-1).mean()

                # plot
                #plt.plot(theta[0,:self.n_attributes].cpu().detach().numpy().T)
                #plt.show()
                # plot
                # no axis labels
                axs[i,beta_idx].axis("off")
                axs[i,beta_idx].plot(theta[0,:self.n_attributes].cpu().detach().numpy().T)

          # label rows and columns
        for i in range(T):
            axs[i,0].set_ylabel(f"t={t[i]:.2f}")
        for i in range(BETAS):
            axs[0,i].set_title(f"beta={betas[i]}")
        plt.title("All attributes")
        plt.show()
        print("done")

              
        # plot entropies
        fig, ax = plt.subplots(1,1,figsize=(12,8))
        for i in range(BETAS):
            ax.plot(entropies[:,i].cpu().detach().numpy(),label=f"beta={betas[i]}")
        # labels
        plt.title("Entropy (all attributes)")
        plt.legend()
        plt.show()


        for attr_idx, attr_name in enumerate(self.tokenizer.note_attribute_order):
            attr_entropies = torch.zeros((T,BETAS))
            # make subplot for each t, each beta
            fig, axs = plt.subplots(T,BETAS,figsize=(12,8))

            for beta_idx, beta in enumerate(betas):
                for i in range(T):
                    theta = self.step(batch,0, tmp_beta1=beta,tmp_t=t[i], preview_input_dist=True)
                    # only consider attribute
                    attr_theta = theta[0,attr_idx::self.n_attributes]
                    # calculate entropy
                    attr_entropies[i,beta_idx] = -torch.sum(attr_theta * torch.log(attr_theta + 1e-12),dim=-1).mean()

                    axs[i,beta_idx].axis("off")
                    axs[i,beta_idx].plot(theta[0,attr_idx].cpu().detach().numpy().T)
            for i in range(T):
                axs[i,0].set_ylabel(f"t={t[i]:.2f}")
            for i in range(BETAS):
                axs[0,i].set_title(f"beta={betas[i]}")
            plt.title(f"Attribute {attr_name}")
            plt.show()
            print("done")

            # plot entropies
            fig, ax = plt.subplots(1,1,figsize=(12,8))
            for i in range(BETAS):
                ax.plot(attr_entropies[:,i].cpu().detach().numpy(),label=f"beta={betas[i]}")
            # labels
            plt.title(f"Entropy of attribute {attr_name}")
            plt.legend()
            plt.show()


    def step(self, batch, batch_idx, tmp_beta1=None, preview_input_dist=False, tmp_t=None):
        if self.one_hot_input:
            x = batch
        else:
            x = torch.nn.functional.one_hot(batch, num_classes=len(self.vocab)).float()

        if tmp_beta1 is not None:
            beta1 = tmp_beta1
        else:
            beta1 = self.beta1
        if tmp_t is not None:
            t = tmp_t
        else:
            t = torch.rand((x.shape[0], 1, 1), device=x.device)

        K = x.shape[-1] 
        # get one t in [0, 1] per sample in batch

        beta = beta1 * (t**2)
        # na_k = format_mask.sum(dim=-1, keepdim=True)
        na_k = K
        y_mean = beta * (na_k * x - 1)
        y_std = (beta * na_k).sqrt()
        y = torch.randn(x.shape, device=x.device) * y_std + y_mean
        # if format mask is not 1 set to -ifn
        theta = torch.nn.functional.softmax(y, dim=-1)

        if self.vocab_theta:
            # multiply with format mask
            theta = theta * self.format_mask[None, ...].to(theta.device)
            # normalize
            theta = theta / theta.sum(dim=-1, keepdim=True)

        if preview_input_dist:
            return theta

        output_probs = self.discrete_output_distribution(theta, t)
        ehat = output_probs
        loss = na_k * beta1 * t * ((x - ehat) ** 2)

        metrics = {}
        metrics["l_inf_loss"] = loss.mean()
        return metrics

    def sample(
        self,
        mask=None,
        batch_size=1,
        nb_steps=10,
        device="cpu",
        eps_=1e-10,
        plot_interval=-1,
        argmax=False,
    ):
        self.eval()

        theta = torch.ones(
            (batch_size, self.n_events * self.n_attributes, len(self.vocab)),
            device=device,
        )  

        if self.vocab_theta:
            theta = theta * self.format_mask[None, ...].to(theta.device)
            # turn into uniform prior
            theta = theta / theta.sum(dim=-1, keepdim=True)

        # mult with mask
        if mask is not None:
            mask = mask[None, ...].to(theta.device)
            theta = theta * mask
            # normalize
            theta = theta / theta.sum(dim=-1, keepdim=True)

        K = theta.shape[-1]

        entropies = []
    
        first_event_probs = []
        first_event_thetas = []

        first_event_thetas.append(theta[0, : self.n_attributes, :].detach().cpu())


        for i in tqdm(range(1, nb_steps + 1)):
            t = (i - 1) / nb_steps
            t = t * torch.ones(
                (theta.shape[0], 1, 1), device=theta.device, dtype=theta.dtype
            )

            k_probs = self.discrete_output_distribution(theta, t)  # (B, D, K)
            if plot_interval > 0 and i % plot_interval == 0:
                # plt.imshow(k_probs[0].cpu().detach().numpy().T, aspect="auto", interpolation="none")
                # plt.show()
                # create subplot for each attribute
                fig, axs = plt.subplots(self.n_attributes, 1, figsize=(12, 8))
                for j in range(self.n_attributes):
                    # axs[j].plot(k_probs[0, j].cpu().detach().numpy().T)
                    # bar plot
                    axs[j].bar(range(K), k_probs[0, j].cpu().detach().numpy().T)
                plt.show()

            k = torch.distributions.Categorical(probs=k_probs).sample()  # (B, D)
            print(k.shape)
            # assert k.shape == k_probs
            alpha = self.beta1 * (2 * i - 1) / (nb_steps**2)

            e_k = F.one_hot(k, num_classes=K).float()  # (B, D, K)
            mean = alpha * (K * e_k - 1)
            var = alpha * K
            std = torch.full_like(mean, fill_value=var).sqrt()
            eps = torch.randn_like(e_k)
            y = mean + std * eps  # (B, D, K)

            theta = F.softmax(y + torch.log(theta + eps_), dim=-1)

            if mask is not None:
                theta = theta * mask
                # normalize
                theta = theta / theta.sum(dim=-1, keepdim=True)

            if self.vocab_theta:
                theta = theta * self.format_mask[None, ...].to(theta.device)
                theta = theta / theta.sum(dim=-1, keepdim=True)

            # add theta entropy
            entropies.append(-torch.sum(k_probs * torch.log(k_probs + eps_), dim=-1).mean().detach().cpu())
            first_event_probs.append(k_probs[0, : self.n_attributes, :].detach().cpu())

            first_event_thetas.append(theta[0, : self.n_attributes, :].detach().cpu())

        k_probs_final = self.discrete_output_distribution(theta, torch.ones_like(t))

        if argmax:
            k_final = k_probs_final.argmax(dim=-1)
        else:
            k_final = torch.distributions.Categorical(probs=k_probs_final).sample()

        # plot entropies
        plt.plot(entropies)
        plt.show()

        # plot first event probs
        first_event_probs = torch.stack(first_event_probs, dim=0)

        # first_event_probs = torch.log(first_event_probs + eps_)

        first_event_thetas = torch.stack(first_event_thetas, dim=0)


        for i in range(self.n_attributes):
            # plot first event thetas
            plt.imshow(first_event_thetas[:,i].T, aspect="auto", interpolation="none")
            plt.title(f"Thetas for attribute {self.tokenizer.note_attribute_order[i]}")
            plt.show()

            plt.imshow(first_event_probs[:,i].T, aspect="auto", interpolation="none")
            plt.title(f"Probs for attribute {self.tokenizer.note_attribute_order[i]}")
            plt.show()



        return k_final

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
        loss = metrics["l_inf_loss"]
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
        loss = metrics["l_inf_loss"]
        self.log(
            "val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        # learning rate decay
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=self.learning_rate_gamma, step_size=1
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

    model = BFNModel(
        hidden_size=512,
        n_heads=4,
        feed_forward_size=2 * 512,
        n_layers=6,
        vocab=tokenizer.vocab,
        max_seq_len=tokenizer.total_len,
        learning_rate=2e-5,
        tokenizer_config=tokenizer_config,
        learning_rate_gamma=0.99,
        norm_first=True,
        beta1=0.3,
        vocab_theta=False,
    )

    wandb_logger = WandbLogger(log_model="all", project="bfn")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

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
    )

    trainer.fit(
                model,
                trn_dl, 
                val_dl,
    )
                # ckpt_path="checkpoints/upbeat-dawn-53/epoch=14-step=24330-val/loss_epoch=0.00273.ckpt")