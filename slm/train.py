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

class EncoderOnlyModel(pl.LightningModule):
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
        x_bias = -1e5,
        fix_x_bias = False,
        embedding_bias = False,
        standard_mlm_forward=False,
        standard_mlm_masking=False,
        avg_positional_encoding = False,
        use_positional_encoding = False,
        mlm_restricted_sampling = True,
        enforce_constraint_in_forward = True,
        neighbour_superposition = False
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
        self.positional_encoding  = nn.Parameter(torch.zeros(1, max_seq_len*2, hidden_size), requires_grad=True)
        self.embedding_layer = nn.Linear(vocab_size, hidden_size, bias=embedding_bias)
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

        self.x_bias = x_bias
        self.fix_x_bias = fix_x_bias

        self.decoder_output_layer = nn.Linear(hidden_size, vocab_size)
        
        self.seq_len = max_seq_len

        self.n_attributes = len(self.tokenizer.note_attribute_order)

        self.normalize_by_masking_ratio = normalize_by_masking_ratio

        self.learning_rate_gamma = learning_rate_gamma

        self.standard_mlm_forward = standard_mlm_forward
        self.standard_mlm_masking = standard_mlm_masking

        if self.standard_mlm_forward:
            self.attribute_mask_tokens = nn.Parameter(torch.ones(1, self.n_attributes, hidden_size), requires_grad=True)

        self.avg_positional_encoding = avg_positional_encoding
        self.use_positional_encoding = use_positional_encoding
        
        self.mlm_restricted_sampling = mlm_restricted_sampling

        self.enforce_constraint_in_forward = enforce_constraint_in_forward

        self.neighbour_superposition = neighbour_superposition

    def mlm_forward(self, x):
        format_mask = self.format_mask[None, ...].to(x.device)
        xin = x

        x = x * format_mask

        # figure out if token is masked, i.e more than one token is masked

        x = einops.rearrange(x, "b (t a) v -> b t a v", a=self.n_attributes)

        is_masked = x.sum(-1) > 1

        ze = self.embedding_layer(x)
        
        # replace masked attribute embeddings with attribute mask tokens
        ze  = ~is_masked[...,None] * ze + is_masked[...,None] * self.attribute_mask_tokens[None, ...].to(self.device)

        # sum across attributes
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

        decoder_logits = einops.rearrange(
            decoder_logits, "b t a v -> b (t a) v"
        )

        format_mask_expanded = format_mask * torch.ones_like(xin, device=self.device)
        decoder_logits[format_mask_expanded<0.5] = self.x_bias

        if not self.training and self.mlm_restricted_sampling:
            decoder_logits[xin<0.5] = self.x_bias

        # crop to decoder length
        return decoder_logits


    def forward(self, x):
        if self.standard_mlm_forward:
            return self.mlm_forward(x)

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

        if self.enforce_constraint_in_forward:
            # force logits to respect constraint
            if self.fix_x_bias:
                decoder_logits[x<0.5] = self.x_bias
            else:
                decoder_logits = decoder_logits + self.x_bias * (1-x)
    
            decoder_logits = einops.rearrange(decoder_logits, "b t a v -> b (t a) v", a=self.n_attributes)
        
        else:
            decoder_logits = einops.rearrange(
                decoder_logits, "b t a v -> b (t a) v", a=self.n_attributes
            )
            decoder_logits[
                (format_mask * torch.ones_like(decoder_logits, device=self.device)) < 0.5
            ] = self.x_bias

        # crop to decoder length
        return decoder_logits

    def generate_batch(self, x, temperature=1, top_p=1, top_k=0, fixed_order=False):

        self.eval()

        with torch.no_grad():

            x = x * self.format_mask[None, ...].to(x.device)

            batch, time_attr, ft = x.shape

            # sampling order
            order = torch.randperm(time_attr)

            for i in tqdm(order):

                logits = self(x.float())

                curr_logits = logits[:, i]

                curr_x = x[:, i]


                # invert probs
                # flatten

                curr_logits = top_k_top_p_filtering(curr_logits, top_k=top_k, top_p=top_p)

                curr_probs = F.softmax(curr_logits / temperature, dim=-1)

                curr_probs[curr_x < 0.5] = 0

                # renormalize
                curr_probs = curr_probs / curr_probs.sum(dim=-1, keepdim=True)

                curr_sampled = torch.multinomial(curr_probs, 1).squeeze(-1)

                # convert to one hot
                curr_one_hot = torch.nn.functional.one_hot(
                    curr_sampled, num_classes=curr_logits.shape[-1]
                ).float()

                x[:, i] = curr_one_hot

        return x
    
    def generate(self, x, sampling_steps=None, temperature=1, top_p=1, top_k=0, fixed_order=False):
        if sampling_steps is None:
            sampling_steps = self.tokenizer.config["max_notes"]*len(self.tokenizer.note_attribute_order)
        self.eval()
        with torch.no_grad():
            x = x
            # multiply by format mask
            x = x * self.format_mask[None, ...].to(x.device)
            batch, time_attr, ft = x.shape
            # linspace from schedule
            schedule = (
                torch.linspace(0, 1, sampling_steps, dtype=torch.float32, device=x.device)
            )
            schedule = schedule
            total_tokens = time_attr
            # count number of known tokens
            masked_tokens = (x.sum(-1) > 1).sum().int()
            # find masking ratio
            masking_ratio = masked_tokens / total_tokens

            # find step in schedule
            step = torch.argmin((1 - schedule - masking_ratio).abs())

            for i in tqdm(range(step + 1, sampling_steps)):

                logits = self(x.float())

                logits[x<0.5] = self.x_bias

                # invert probs
                # flatten
                flat_logits = einops.rearrange(logits, "b ta v -> (b ta) v")

                flat_logits = top_k_top_p_filtering(flat_logits, top_k=top_k, top_p=top_p)

                flat_probs = F.softmax(flat_logits / temperature, dim=-1)

                sampled = torch.multinomial(flat_probs, 1).squeeze(-1)


                flat_x = einops.rearrange(x, "b ta v -> (b ta) v")

                masked_indices = torch.where(flat_x.sum(-1) > 1)[0]
                #
                n_masked = masked_indices.shape[0]
                target_masking_ratio = 1 - schedule[i].item()

                target_n_masked = int(total_tokens * target_masking_ratio)
                # tokens to unmask

                n_tokens_to_unmask = n_masked - target_n_masked

                # get indices of tokens to unmask
                # get indices of masked tokens
                # shuffle masked indices
                if not fixed_order:
                    masked_indices = masked_indices[torch.randperm(n_masked)]
                indices_to_unmask = masked_indices[:n_tokens_to_unmask]

                # replace with sampled values
                flat_x[indices_to_unmask] = torch.nn.functional.one_hot(
                    sampled[indices_to_unmask], num_classes=flat_x.shape[-1]
                ).float()

                # plot

                x = einops.rearrange(
                    flat_x, "(b ta) v -> b ta v", b=batch, ta=time_attr
                )
        return x
    
    def compute_perplexity(self, x, tgt):

        # assert that x is not batched
        assert len(x.shape) == 2
        self.eval()
        x = x.clone()
        tgt = tgt.clone()

        # get unkown position indices
        unkown_positions = x[0].sum(dim=-1) > 1
        
        log_probabilities = []
        with torch.no_grad():
            for i in tqdm(unkown_positions):
                logits = self(x)
                logits[x<0.5] = self.x_bias
                probs = F.softmax(logits, dim=-1)
                probs = probs * x # set non possible tokens to 0
                probs = probs * self.format_mask[None, ...].to(x.device)
                # normalize
                probs = probs / probs.sum(dim=-1, keepdim=True)
                # get probabilities
                current_prob = (probs[:, i] * tgt[:, i]).sum(dim=-1)
                current_log_prob = torch.log(current_prob)
                log_probabilities.append(current_log_prob)
                # replace with target
                x[:, i ] = tgt[:, i]
            total_prob = torch.stack(log_probabilities).sum(dim=0)
            return total_prob

    def step(self, batch, batch_idx):
        if self.one_hot_input:
            x = batch
        else:
            x = torch.nn.functional.one_hot(batch, num_classes=len(self.vocab)).float()
        
        batch_size = x.shape[0]

        if self.standard_mlm_forward:
            
            # create a binary mask of size (batch_size, (notes attributes))
            masking_probs = torch.rand(batch_size, device=self.device)
            mask = (
                torch.rand((x.shape[0], x.shape[1]), device=self.device)
                < masking_probs[:, None]
            )
            mask = mask[:,:,None] * torch.ones_like(x, device=self.device)

        else:                

            masking_probs = torch.rand(batch_size, device=self.device)
            position_mask = (
                torch.rand((x.shape[0], x.shape[1]), device=self.device)
                < masking_probs[:, None]
            )

            # create masking ratios
            superposition_probs = torch.rand(batch_size, device=self.device)
            superposition = torch.rand_like(x, device=self.device)<superposition_probs[:,None,None]

            mask = position_mask[:,:,None] * superposition


            if self.neighbour_superposition:

                neighbour_sum_superposition = x.sum(dim=1, keepdim=True) > 1

                # get indices of tokens to apply neighbour sum mask

                neighbour_mask_probs = torch.rand(batch_size, device=self.device)

                neighbour_position_mask = (
                    torch.rand((x.shape[0], x.shape[1]), device=self.device)
                    < neighbour_mask_probs[...,None]
                )

                neighbour_superposition_mask = neighbour_position_mask[:,:,None] * neighbour_sum_superposition

                # multiply by position mask
                neighbour_sp = neighbour_superposition_mask * position_mask[:,:,None]

                mask = mask + neighbour_sp

                mask = torch.clamp(mask, 0, 1)

    
        masked_x = torch.clamp(x + mask, 0, 1)

        # multiply by format mask
        masked_x = masked_x * self.format_mask[None, ...].to(masked_x.device)

        target = x
        logits = self(masked_x)

        target_idx = torch.argmax(target, dim=-1)

        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_idx.reshape(-1),
            reduction = "none",
        )

        known_positions = (masked_x.sum(dim=-1) == 1).flatten()

        ce[known_positions] = 0

        # reshape to batch, loss
        ce = ce.reshape(batch_size, -1)

        norm_ce = (ce.mean(dim=-1) / masking_probs).mean()

        metrics = {}
        metrics["cross_entropy"] = ce.mean()
        metrics["cross_entropy_normalized"] = norm_ce
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
        if self.normalize_by_masking_ratio:
            loss = metrics["cross_entropy_normalized"]
        else:
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
        if self.normalize_by_masking_ratio:
            loss = metrics["cross_entropy_normalized"]
        else:
            loss = metrics["cross_entropy"]
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = pl.pytorch.utilities.grad_norm(self.layer, norm_type=2)
    #     self.log_dict(norms)

    def configure_optimizers(self):
        # learning rate decay
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        # add 1 epoch linear warmup
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
        #                                               lambda epoch: max(0.1, self.learning_rate_gamma ** epoch))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.learning_rate_gamma, step_size=1)
        # scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=1,
        #     max_epochs=100,
        #     warmup_start_lr=0.0,
        #     eta_min=1e-6,
        #     last_epoch=-1,
        # )
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

    trn_ds = MidiDataset(
        cache_path="./artefacts/trn_midi_records_unique_pr.pt",
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

    USE_MLM_BASELINE = True
    
    model = EncoderOnlyModel(
        hidden_size=768,
        n_heads=12,
        feed_forward_size=4 * 768,
        n_layers=12,
        vocab=tokenizer.vocab,
        max_seq_len=tokenizer.total_len,
        learning_rate=1e-4 if not USE_MLM_BASELINE else 1e-4,
        tokenizer_config=tokenizer_config,
        normalize_by_masking_ratio=False,
        learning_rate_gamma=0.99,
        norm_first=True,
        x_bias=-1e9,
        fix_x_bias=True,
        embedding_bias=False,
        standard_mlm_forward= USE_MLM_BASELINE,
        standard_mlm_masking= USE_MLM_BASELINE,
        avg_positional_encoding=False,
        use_positional_encoding=False,
        enforce_constraint_in_forward=True,
        neighbour_superposition = False
    )

    wandb_logger = WandbLogger(log_model="all", project="slm")
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    progress_bar_callback = RichProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
    strategy="ddp_find_unused_parameters_true",
    accelerator="gpu",
    devices=[2,3],
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
        # ckpt_path="checkpoints/trim-water-280/epoch=132-step=191919-val/loss_epoch=0.14.ckpt",
        # ckpt_path="checkpoints/trim-water-280/epoch=132-step=191919-val/loss_epoch=0.14.ckpt",
        # ckpt_path="checkpoints/trim-water-280/epoch=132-step=191919-val/loss_epoch=0.14.ckpt"
        # ckpt_path="checkpoints/clear-terrain-265/epoch=111-step=161616-val/loss_epoch=0.14.ckpt"
    )