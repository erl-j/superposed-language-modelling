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
from simplex_diffusion import SimplexDiffusionModel

class BFNModel(pl.LightningModule):
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
        normalize_by_masking_ratio=False,
        learning_rate_gamma=0.9,
        norm_first=False,
        beta1=1.0,
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
        self.format_mask = torch.Tensor(self.tokenizer.get_format_mask())
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

        self.embedding_layer = nn.Linear(vocab_size, hidden_size, bias=False)
        self.decoder_output_layer = nn.Linear(hidden_size, vocab_size, bias=output_bias)

        # doesnt work
        self.tied_embeddings = tied_embeddings
        if tied_embeddings:
            raise NotImplementedError("Tied embeddings not implemented")
            # self.embedding_layer.weight = self.decoder_output_layer.weight.T
        
        self.n_attributes = len(self.tokenizer.note_attribute_order)
        self.n_events = self.tokenizer.config["max_notes"]
        self.beta1 = beta1
        self.vocab_theta = vocab_theta
        self.warmup_steps = warmup_steps
        self.annealing_steps = annealing_steps
        self.min_lr_ratio = min_lr_ratio
        self.learning_rate = learning_rate
        self.learning_rate_gamma = learning_rate_gamma

        self.fourier_t_embedding = fourier_t_embedding
        if fourier_t_embedding:
            self.time_embedder = nn.Sequential(GaussianFourierProjection(embedding_dim=hidden_size),nn.Linear(hidden_size, hidden_size))
        else:
            self.time_embedder = torch.nn.Linear(1, hidden_size)
        
    def forward(self, theta, t):
        theta = theta.clone()
        t = t.clone()

        if self.fourier_t_embedding:
            t_z = self.time_embedder(t)
        else:
            t_z = self.time_embedder(t[:,None])

        format_mask = self.format_mask[None, ...].to(theta.device)
        theta[format_mask.expand_as(theta)<0.5] = 0
        # renormalize
        theta = theta / theta.sum(dim=-1, keepdim=True)
        x = theta

        x = einops.rearrange(x, "b (t a) v -> b t a v", a=self.n_attributes)
        ze = self.embedding_layer(x)
        ze += t_z
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
        decoder_logits[format_mask.expand_as(decoder_logits) <0.5] = -1e12
        return decoder_logits

    def discrete_output_distribution(self, theta, t, temperature=1.0, top_p=1.0, top_k=0):
        output_dist = self.forward(theta, t)
        if top_k > 0 or top_p < 1.0:
            output_dist = einops.rearrange(output_dist, "b s v -> (b s) v")
            output_dist = top_k_top_p_filtering(output_dist, top_k=top_k, top_p=top_p)
            output_dist = einops.rearrange(output_dist, "(b s) v -> b s v", s=self.n_attributes * self.n_events)

        output_probs = torch.nn.functional.softmax(output_dist/temperature, dim=-1)
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
        betas = [0.03,0.04,0.05,0.065,0.075,0.1,0.2,0.5]
        T = 10

        # linspace from 0 to 1 and includes 0 and 1
        t = torch.linspace(0,1,T) 
        assert t[0] == 0 # append 0
        assert t[-1] == 1

        # append 1
        BETAS = len(betas)
        
        entropies = torch.zeros((T, BETAS, self.n_attributes))
        for beta_idx, beta in enumerate(betas):
            for i in range(T):
                theta = self.step(batch.clone(),0, tmp_beta1=beta,tmp_t=t[i], preview_input_dist=True)

                # multiply with format mask
                theta = theta * self.format_mask[None, ...].to(theta.device)

                # renormalize
                theta = theta / theta.sum(dim=-1, keepdim=True)
                
                x1h = torch.nn.functional.one_hot(batch, num_classes=len(self.vocab)).float()
                # calculate entropy
                entropy = -torch.sum(x1h * torch.log(theta+1e-12),dim=-1)

                entropy = einops.rearrange(entropy, "b (n a) -> b n a", a=self.n_attributes).mean(dim=-2).mean(0)

                entropies[i, beta_idx,:] = entropy 

        # for each beta, plot entropy over t for all attributes
        for beta_idx, beta in enumerate(betas):
            for i in range(self.n_attributes):
                plt.plot(t, entropies[:,beta_idx,i].detach().cpu().numpy(), label=f"{self.tokenizer.note_attribute_order[i]}")
            plt.title(f"Beta={beta}")
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
        prior,
        batch_size=1,
        nb_steps=10,
        temperature=1.0,
        device="cpu",
        eps_=1e-10,
        plot_interval=-1,
        argmax=False,
        top_k=0,
        top_p=0.0,
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
        if prior is not None:
            prior = prior[None, ...].to(theta.device)
            theta = theta * prior
            # normalize
            theta = theta / theta.sum(dim=-1, keepdim=True)

        K = theta.shape[-1]

        entropies = []
    
        first_event_probs = []
        first_event_thetas = []

        first_event_thetas.append(theta[0, : self.n_attributes, :].detach().cpu())

        k = torch.zeros((batch_size, self.n_events * self.n_attributes), device=device, dtype=torch.long)
        for i in tqdm(range(1, nb_steps + 2)):
            t = (i - 1) / nb_steps
            t = t * torch.ones(
                (theta.shape[0], 1, 1), device=theta.device, dtype=theta.dtype
            )

            k_probs = self.discrete_output_distribution(theta, t, temperature=temperature, top_k=top_k, top_p=top_p)
            # if plot_interval > 0 and i % plot_interval == 0:
            #     # plt.imshow(k_probs[0].cpu().detach().numpy().T, aspect="auto", interpolation="none")
            #     # plt.show()
            #     # create subplot for each attribute
            #     fig, axs = plt.subplots(self.n_attributes, 1, figsize=(12, 8))
            #     for j in range(self.n_attributes):
            #         # axs[j].plot(k_probs[0, j].cpu().detach().numpy().T)
            #         # bar plot
            #         axs[j].bar(range(K), k_probs[0, j].cpu().detach().numpy().T)
            #     plt.show()

            k_new = torch.distributions.Categorical(probs=k_probs).sample()  # (B, D)
            k = k_new
            # if (k_new == k).all():
            #     print(f"Converged at step {i}")
            #     k = k_new
            #     break
                
            # else:
            #     # count differences
            #     diff = (k_new != k).sum()
            #     print(f"Step {i}, {diff} differences")
            #     k = k_new
            
            # assert k.shape == k_probs
            alpha = self.beta1 * (2 * i - 1) / (nb_steps**2)

            e_k = F.one_hot(k, num_classes=K).float()  # (B, D, K)
            mean = alpha * (K * e_k - 1)
            var = alpha * K
            std = torch.full_like(mean, fill_value=var).sqrt()
            eps = torch.randn_like(e_k)
            y = mean + std * eps  # (B, D, K)

            theta = F.softmax(y + torch.log(theta + eps_), dim=-1)

            if prior is not None:
                theta = theta * prior * self.format_mask[None, ...].to(theta.device)
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

        k_final = k_probs_final.argmax(dim=-1)

        if plot_interval > 0:
            # plot entropies
            plt.plot(entropies)
            # add horizontal line at 0
            plt.axhline(0, color="black", linestyle="--")
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

    # def configure_optimizers(self):
    #     # learning rate decay
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    #     scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, gamma=self.learning_rate_gamma, step_size=1
    #     )
    #     return [optimizer], [scheduler]
    
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
    
    def initialize_from_different_model(self, src_model):
        for token in self.vocab:
            if token not in src_model.vocab:
                print(f"Token {token} not found in source model")
                continue
            else:
                print(f"Transplanting token {token}")
                src_idx = src_model.vocab.index(token)
                tgt_idx = self.vocab.index(token)
                self.embedding_layer.weight.data[:,tgt_idx] = src_model.embedding_layer.weight.data[:,src_idx]
                self.decoder_output_layer.weight.data[tgt_idx,:] = src_model.decoder_output_layer.weight.data[src_idx,:]
        if self.tied_embeddings:
            raise NotImplementedError("Tied embeddings not implemented")
            # self.embedding_layer.weight = self.decoder_output_layer.weight.T


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

    # model = BFNModel(
    #     hidden_size=768,
    #     n_heads=12,
    #     feed_forward_size=2 * 768,
    #     n_layers=12,
    #     vocab=tokenizer.vocab,
    #     learning_rate=5e-5,
    #     tokenizer_config=tokenizer_config,
    #     learning_rate_gamma=0.99,
    #     norm_first=True,
    #     beta1=0.1,
    #     vocab_theta=False,
    #     warmup_steps=1_000,
    #     annealing_steps=200_000,
    #     min_lr_ratio=0.1,
    #     tied_embeddings=False,
    #     fourier_t_embedding=True,
    #     output_bias=False,
    #     use_adamw=False,
    # )

    model = BFNModel.load_from_checkpoint(
                checkpoint_path = "./checkpoints/polished-dream-124/last.ckpt",
                map_location="cpu",
                learning_rate=2e-5,
                annaling_steps=200_000*100,
                warmup_steps=5,
                min_lr_ratio=0.1,
    )

    # set global step to annealing steps
    # set global step manually
    # src_model = SimplexDiffusionModel.load_from_checkpoint(
    #             checkpoint_path = "./checkpoints/flowing-paper-64/last.ckpt",
    #             map_location="cpu"
    # )
    # model.initialize_from_different_model(src_model)

    format_mask = torch.Tensor(tokenizer.get_format_mask())
    
    dummy_sample = format_mask.argmax(-1)

    model.step(dummy_sample[None, :].repeat(BATCH_SIZE, 1),0)

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

    wandb_logger = WandbLogger(log_model="all", project="bfn")
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
        # gradient_clip_val=1.0,
    )

    trainer.fit(
                model,
                trn_dl, 
                val_dl,
    )
                