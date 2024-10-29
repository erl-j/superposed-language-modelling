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
import einops
from tqdm import tqdm
from util import top_k_top_p_filtering


def random_add_masking(x):
    batch_size = x.shape[0]               
    masking_probs = torch.rand(batch_size, device=x.device)
    position_mask = (
        torch.rand((x.shape[0], x.shape[1]), device=x.device)
        < masking_probs[:, None]
    )
    # create masking ratios
    superposition_probs = torch.rand(batch_size, device=x.device)
    superposition = torch.rand_like(x, device=x.device)<superposition_probs[:,None,None]
    mask = position_mask[:,:,None] * superposition
    masked_x = torch.clamp(x + mask, 0, 1)
    return masked_x
    

class SuperposedLanguageModel(pl.LightningModule):
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
        learning_rate_gamma=0.9,
        norm_first=False,
        enforce_constraint_in_forward = True,
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
        self.positional_encoding  = nn.Parameter(torch.zeros(1, max_seq_len*2, hidden_size), requires_grad=True)
        self.embedding_layer = nn.Linear(vocab_size, hidden_size, bias=False)
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
        self.enforce_constraint_in_forward = enforce_constraint_in_forward

    def convert_to_half(self):
        return self.half()

    def forward(self, x):

        format_mask = self.format_mask[None, ...].to(x.device).to(x.dtype)
        x = x * format_mask

        x = einops.rearrange(x, "b (t a) v -> b t a v", a=self.n_attributes)

        ze = self.embedding_layer(x)
        ze = ze.sum(dim=2)
        zl = self.transformer(ze)
        note_z = einops.rearrange(zl, "b t ft -> b t 1 ft")
        note_z = note_z.repeat(1, 1, self.n_attributes, 1)
        decoder_logits = self.decoder_output_layer(note_z)
        if self.enforce_constraint_in_forward:
            # force logits to respect constraint
            # decoder_logits[x.isclose(torch.zeros_like(x,dtype=x.type))] = torch.finfo(x.dtype).min
            # # decoder_logits[x.isclose(torch.ones_like(x,dtype=x.type))] = torch.finfo(x.dtype).max
            # decoder_logits = einops.rearrange(decoder_logits, "b t a v -> b (t a) v", a=self.n_attributes)
            # decoder_logits[
            #     (format_mask * torch.ones_like(decoder_logits, device=self.device)).isclose(torch.zeros_like(decoder_logits))
            # ] = torch.finfo(x.dtype).min

            decoder_logits[x==0] = torch.finfo(x.dtype).min
            # decoder_logits[x.isclose(torch.ones_like(x,dtype=x.type))] = torch.finfo(x.dtype).max
            decoder_logits = einops.rearrange(decoder_logits, "b t a v -> b (t a) v", a=self.n_attributes)
            decoder_logits[
                (format_mask * torch.ones_like(decoder_logits, device=self.device))==0
            ] = torch.finfo(x.dtype).min
        # crop to decoder length
        return decoder_logits

    @torch.inference_mode()
    def generate(
        self,
        x,
        sampling_steps=None,
        temperature=1,
        top_p=1,
        top_k=0,
        order="random",
        attribute_temperature=None,
        tokens_per_step=1,
    ):
        if sampling_steps is None:
            sampling_steps = self.tokenizer.config["max_notes"]*len(self.tokenizer.note_attribute_order)
        self.eval()
        dtype = self.embedding_layer.weight.dtype
        # convert to model dtype, (fp32, fp16)
        x = x.to(dtype)
        with torch.no_grad():
            x = x
            # multiply by format mask
            x = x * self.format_mask[None, ...].to(x.device).to(dtype)
            x = self.tokenizer.collapse_undefined_attributes(x)
            batch, time_attr, ft = x.shape
            # count number of known tokens
            masked_tokens = (x.sum(-1) > 1).sum().int().item()
            with tqdm(total=masked_tokens) as pbar:
                while True:
                    masked_tokens_before = (x.sum(-1) > 1).sum().int().item()
                    # take time of forward pass
                    logits = self(x)
                    # invert probs
                    # flatten
                    flat_logits = einops.rearrange(logits, "b ta v -> (b ta) v")
                    flat_logits = top_k_top_p_filtering(flat_logits, top_k=top_k, top_p=top_p)
                    t = temperature
                    if attribute_temperature is not None:
                        # turn t into 1,1,a tensor
                        attr_t = torch.ones(self.n_attributes, device=x.device) * temperature
                        for k, v in attribute_temperature.items():
                            # get attribute index
                            attr_idx = self.tokenizer.note_attribute_order.index(k)
                            attr_t[attr_idx] = v
                        attr_t = einops.repeat(attr_t, "a -> (b e a) 1", e=self.tokenizer.config["max_notes"], b=batch)
                        t = attr_t
                    flat_probs = F.softmax(flat_logits / t, dim=-1)
                    flat_x = einops.rearrange(x, "b ta v -> (b ta) v")
                    # renormalize
                    flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(flat_probs, 1).squeeze(-1)
                    flat_x = einops.rearrange(x, "b ta v -> (b ta) v")
                    masked_indices = torch.where(flat_x.sum(-1) > 1)[0]
                    n_masked = masked_indices.shape[0]
                    n_tokens_to_unmask = tokens_per_step
                    if order == "random":
                        masked_indices = masked_indices[torch.randperm(n_masked)]
                        indices_to_unmask = masked_indices[:n_tokens_to_unmask]
                    elif order == "attribute":
                        indices_to_unmask = masked_indices[:n_tokens_to_unmask]
                    elif order == "lowest_entropy":
                        masked_probs = flat_probs[masked_indices]
                        entropy = -torch.sum(masked_probs * torch.log(masked_probs), dim=-1)
                        masked_indices = masked_indices[torch.argsort(entropy)]
                        indices_to_unmask = masked_indices[:n_tokens_to_unmask]       
                    elif order == "highest_entropy":
                        masked_probs = flat_probs[masked_indices]
                        entropy = -torch.sum(masked_probs * torch.log(masked_probs), dim=-1)
                        masked_indices = masked_indices[torch.argsort(entropy, descending=True)]
                        indices_to_unmask = masked_indices[:n_tokens_to_unmask] 

                    # replace with sampled values
                    flat_x[indices_to_unmask] = torch.nn.functional.one_hot(
                        sampled[indices_to_unmask], num_classes=flat_x.shape[-1]
                    ).to(dtype)
                    # plot

                    x = einops.rearrange(
                        flat_x, "(b ta) v -> b ta v", b=batch, ta=time_attr
                    )

                    x = self.tokenizer.collapse_undefined_attributes(x)

                    # masekd tokens after
                    masked_tokens_after = (x.sum(-1) > 1).sum().int().item()

                    pbar.update(masked_tokens_before - masked_tokens_after)
                    if masked_tokens_after == 0:
                        break
        return x
    
    def step(self, batch, batch_idx):

        x = torch.nn.functional.one_hot(batch, num_classes=len(self.vocab)).float()

        batch_size = x.shape[0]
        masked_x = random_add_masking(x)
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
        metrics = {}
        metrics["cross_entropy"] = ce.mean()
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
        else:
            loss = metrics["cross_entropy"]
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # learning rate decay
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.learning_rate_gamma, step_size=1)
        return [optimizer], [scheduler]

    def initialize_from_different_model(self, src_model, skip_tokens=[]):
        for token in self.vocab:
            if token not in src_model.vocab:
                print(f"Token {token} not found in source model")
                continue
            elif any([token.split(":")[0] in skip for skip in skip_tokens]):
                print(f"Skipping token {token}")
                continue
            else:
                print(f"Transplanting token {token}")
                src_idx = src_model.vocab.index(token)
                tgt_idx = self.vocab.index(token)
                self.embedding_layer.weight.data[:, tgt_idx] = (
                    src_model.embedding_layer.weight.data[:, src_idx]
                )
                self.decoder_output_layer.weight.data[tgt_idx, :] = (
                    src_model.decoder_output_layer.weight.data[src_idx, :]
                )
        # now copy transformer
        self.transformer.load_state_dict(src_model.transformer.state_dict())

if __name__ == "__main__":

    DATASET = "mmd_loops"

    BATCH_SIZE = 40

    tag_list = open(f"./data/{DATASET}/tags.txt").read().splitlines()

    N_BARS = 4 if DATASET == "harmonic" else 4

    tokenizer_config = {
        "ticks_per_beat": 24 if (DATASET == "mmd_loops" or DATASET ==  "harmonic") else 48,
        "time_hierarchy": "tick",
        "use_duration": True,
        "pitch_range": [0, 128],
        "max_beats": 4 * N_BARS,
        "max_notes": 75 * N_BARS if DATASET == "mmd_loops" else 20 * N_BARS,
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

    print(f"Vocab size: {len(tokenizer.vocab)}")

    # print note attribute order
    print(tokenizer.note_attribute_order)

    FINETUNE = False

    if not FINETUNE:

        model = SuperposedLanguageModel(
            hidden_size=768,
            n_heads=12,
            feed_forward_size=3072,
            n_layers=12,
            vocab=tokenizer.vocab,
            max_seq_len=tokenizer_config["max_notes"] * len(tokenizer.note_attribute_order),
            learning_rate=1e-4,
            tokenizer_config=tokenizer_config,
            learning_rate_gamma=0.99,
            norm_first=True,
            enforce_constraint_in_forward=True,
        )

    elif FINETUNE:

        src_model = SuperposedLanguageModel.load_from_checkpoint(
            checkpoint_path="./paper_assets/slm_.ckpt",
            map_location="cpu"
        )

        # model= SuperposedLanguageModel(
        #     hidden_size=src_model.hparams.hidden_size,
        #     n_heads=src_model.hparams.n_heads,
        #     feed_forward_size=src_model.hparams.feed_forward_size,
        #     n_layers=src_model.hparams.n_layers,
        #     vocab=tokenizer.vocab,
        #     max_seq_len=src_model.hparams.max_seq_len,
        #     learning_rate=src_model.hparams.learning_rate*0.1,
        #     tokenizer_config=tokenizer_config,
        #     one_hot_input=src_model.hparams.one_hot_input,
        #     normalize_by_masking_ratio=src_model.hparams.normalize_by_masking_ratio,
        #     learning_rate_gamma=0.9999,
        #     norm_first= src_model.hparams.norm_first,
        #     x_bias = src_model.hparams.x_bias,
        #     fix_x_bias = src_model.hparams.fix_x_bias,
        #     embedding_bias = src_model.hparams.embedding_bias,
        #     standard_mlm_forward=src_model.hparams.standard_mlm_forward,
        #     standard_mlm_masking=src_model.hparams.standard_mlm_masking,
        #     avg_positional_encoding = src_model.hparams.avg_positional_encoding,
        #     use_positional_encoding = src_model.hparams.use_positional_encoding,
        #     mlm_restricted_sampling = src_model.hparams.mlm_restricted_sampling,
        #     enforce_constraint_in_forward = src_model.hparams.enforce_constraint_in_forward,
        #     neighbour_superposition = src_model.hparams.neighbour_superposition
        # )

        # # model.initialize_from_different_model(src_model, 
        # #                                       skip_tokens=["onset/tick", "offset/tick"]
        # #                                       )

    mmd_4bar_filter_fn = lambda x: f"n_bars={N_BARS}" in x

    val_ds = MidiDataset(
        cache_path=f"./data/{DATASET}/val_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn if DATASET == "mmd_loops" else None,
        genre_list=tag_list,
        tokenizer=tokenizer,
        min_notes=8 * N_BARS if DATASET == "mmd_loops" else 4 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
    )
    print(f"Loaded {len(val_ds)} validation records")

    trn_ds = MidiDataset(
        cache_path=f"./data/{DATASET}/trn_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn if DATASET == "mmd_loops" else None,
        genre_list=tag_list,
        tokenizer=tokenizer,
        transposition_range=[-4, 4] if DATASET == "mmd_loops" or DATASET == "harmonic" else None,
        min_notes=8 * N_BARS if DATASET == "mmd_loops" else 4 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
    )
    # print len of dataset
    print(f"Loaded {len(trn_ds)} training records")

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

    wandb_logger = WandbLogger(
        log_model=False, project="slm",
    )
    # get name
    name = wandb_logger.experiment.name

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    progress_bar_callback = RichProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=[3,4],
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
                every_n_epochs=1 if DATASET == "mmd_loops" else 100,
            ),
        ],
        logger=wandb_logger,
        gradient_clip_val=1.0,
        # accumulate_grad_batches=4,
        check_val_every_n_epoch=1 if DATASET == "mmd_loops" else 10,
    )

    trainer.fit(
        model,
        trn_dl,
        val_dl,
    )