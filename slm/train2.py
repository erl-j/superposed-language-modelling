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
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
import math

def attribute_dropout(x, n_attributes, dropout_prob):
    """
    Applies attribute dropout to input tensor x. For each sample and attribute,
    with probability dropout_prob, that attribute is set to all ones across the entire sequence.

    Args:
        x: Input tensor of shape (batch_size, seq_len * n_attributes, vocab_size)
        n_attributes: Number of attributes per position
        dropout_prob: Probability of dropping out an attribute
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1] // n_attributes
    vocab_size = x.shape[2]

    # Reshape to separate attributes dimension
    x = einops.rearrange(x, "b (t a) v -> b t a v", a=n_attributes)

    # Generate dropout mask per sample and attribute
    mask = torch.rand((batch_size, 1, n_attributes), device=x.device) > dropout_prob
    mask = mask.expand(-1, seq_len, -1).unsqueeze(-1)

    # Where mask is False, set to ones, otherwise keep original values
    x = torch.where(mask, x, torch.ones_like(x))

    return einops.rearrange(x, "b t a v -> b (t a) v")

def random_add_masking_mml(x):
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

def random_add_masking_variable_superposition(x):
    batch_size = x.shape[0]
    position_masking_probs = torch.rand(batch_size, device=x.device)
    position_mask = (
        torch.rand((x.shape[0], x.shape[1]), device=x.device) < position_masking_probs[:, None]
    )
    # Instead of per-sample superposition probs, use per-position probs
    superposition_probs = torch.rand((x.shape[0], x.shape[1]), device=x.device)
    superposition = (
        torch.rand_like(x, device=x.device) < superposition_probs[:, :, None]
    )
    mask = position_mask[:, :, None] * superposition
    masked_x = torch.clamp(x + mask, 0, 1)
    return masked_x

def random_add_masking_variable_superposition_ratio(x, format_mask):
    batch_size, seq_len, vocab_size = x.shape
    position_masking_ratios = torch.rand(batch_size, device=x.device)

    position_mask = torch.zeros(
        (batch_size, seq_len), dtype=torch.bool, device=x.device
    )
    for i in range(batch_size):
        n_positions = int(seq_len * position_masking_ratios[i])
        indices = torch.randperm(seq_len, device=x.device)[:n_positions]
        position_mask[i, indices] = True

    superposition_ratios = torch.rand((batch_size, seq_len), device=x.device)
    superposition = torch.zeros_like(x, dtype=torch.bool, device=x.device)

    for i in range(batch_size):
        for j in range(seq_len):
            if position_mask[i, j]:
                valid_vocab = torch.where(format_mask[j] == 1)[0]
                n_vocab = int(len(valid_vocab) * superposition_ratios[i, j])
                vocab_indices = valid_vocab[
                    torch.randperm(len(valid_vocab), device=x.device)[:n_vocab]
                ]
                superposition[i, j, vocab_indices] = True

    masked_x = torch.clamp(x + superposition, 0, 1)
    return masked_x

def mlm_mask(x):
    batch_size = x.shape[0]
    masking_probs = torch.rand(batch_size, device=x.device)
    position_mask = (
        torch.rand((x.shape[0], x.shape[1]), device=x.device)
        < masking_probs[:, None]
    )
    # first channel is masking channel
    mask = position_mask[:,:,None]
    # if position mask is true, set to 0
    masked_x = x * (1 - mask)
    masked_x = torch.cat([mask, masked_x], dim=-1)
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
        normalize_input=False,
        norm_first=False,
        enforce_constraint_in_forward = True,
        token_atoms = [],
        use_composite_unembedding = False,
        pos_embedding_attributes = [],
        base_period = None,
        activation = "relu",
        use_cross_entropy_increase_loss = False,
        use_prior_scaled_ce_loss = False,
        dropout = 0.1,
        use_input_bias = False,
        use_output_bias = True,
        use_embedding_l2_normalization = False,
        use_unembedding_l2_normalization = False,
        use_mlm = False,
        weight_by_loops_in_parent_song = False,
        random_add_masking_type = "mml"
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
        self.embedding_layer = nn.Linear(vocab_size if not use_mlm else vocab_size + 1, hidden_size, bias=use_input_bias)
     
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=feed_forward_size,
            norm_first= norm_first,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            ),
            num_layers=n_layers,
        )
        self.decoder_output_layer = nn.Linear(hidden_size, vocab_size, bias=use_output_bias)
        self.seq_len = max_seq_len
        self.n_attributes = len(self.tokenizer.note_attribute_order)
        self.learning_rate_gamma = learning_rate_gamma
        self.enforce_constraint_in_forward = enforce_constraint_in_forward
        self.normalize_input = normalize_input
        self.use_cross_entropy_increase_loss = use_cross_entropy_increase_loss
        self.use_prior_scaled_ce_loss = use_prior_scaled_ce_loss
        self.weight_by_loops_in_parent_song = weight_by_loops_in_parent_song
        self.random_add_masking_type = random_add_masking_type
        

    def convert_to_half(self):
        return self.half()
    
    def mlm_forward(self, x_masked):
        '''
        x_masked: b, t, v+1
        logits: b, t, v
        '''
        x = einops.rearrange(x_masked, "b (t a) vm -> b t a v", a=self.n_attributes)
        position_mask = x[:,:,0]
        x = self.embedding_layer(x)
        ze = x.sum(dim=1)
        zl = self.transformer(ze)
        note_z = einops.rearrange(zl, "b t ft -> b t 1 ft")
        note_z = note_z.repeat(1, 1, self.n_attributes, 1)
        decoder_logits = self.decoder_output_layer(note_z)
        # now we want to have carry over unmasking. Meaning that unmasked tokens retain their original value
        # and masked tokens get the value of the decoder logits
        x_not_masked = x_masked[:,:,1:]
        x_not_masked[position_mask] = decoder_logits[position_mask]
        return x_not_masked

    def forward(self, x):
        format_mask = self.format_mask[None, ...].to(x.device).to(x.dtype)
        x = x * format_mask

        x = einops.rearrange(x, "b (t a) v -> b t a v", a=self.n_attributes)
        x = x.to(self.device)
        if self.normalize_input:
            x2 = x / x.sum(dim=-1, keepdim=True)
        else:
            x2 = x
        ze = self.embedding_layer(x2)
        ze = ze.sum(dim=2)
        zl = self.transformer(ze)
        note_z = einops.rearrange(zl, "b t ft -> b t 1 ft")
        note_z = note_z.repeat(1, 1, self.n_attributes, 1)
        decoder_logits = self.decoder_output_layer(note_z)
        if self.enforce_constraint_in_forward:
   
            # CHANGED THIS TO torch.finfo(decoder_logits.dtype).min FROM torch.finfo(x.dtype).min
            decoder_logits[x==0] = torch.finfo(decoder_logits.dtype).min
            decoder_logits = einops.rearrange(decoder_logits, "b t a v -> b (t a) v", a=self.n_attributes)
            # CHANGED THIS TO torch.finfo(decoder_logits.dtype).min FROM torch.finfo(x.dtype).min
            decoder_logits[
                (format_mask * torch.ones_like(decoder_logits, device=self.device))==0
            ] = torch.finfo(decoder_logits.dtype).min
        # crop to decoder length
        return decoder_logits
    
    @torch.inference_mode()
    def fast_kill_events(self, x, temperature=1):
        self.eval()
        dtype = self.embedding_layer.weight.dtype
        x = x.to(dtype)

        with torch.no_grad():
            x = x
            x = x * self.format_mask[None, ...].to(x.device).to(dtype)
            x = self.tokenizer.collapse_undefined_attributes(x)
            x = self.tokenizer.sanitize_mask(x, event_indices=range(self.tokenizer.config["max_notes"]))
            logits = self(x)
            logits = logits / temperature
            new_x = self.tokenizer.get_undefined_probs(x, logits)
        return new_x
        
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
            # x = self.tokenizer.normalize_constraint(x
            x = self.tokenizer.collapse_undefined_attributes(x)
            # sanitize all events
            x = self.tokenizer.sanitize_mask(x, event_indices=range(self.tokenizer.config["max_notes"]))
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
                    n_attributes = len(self.tokenizer.note_attribute_order)
                    n_masked = masked_indices.shape[0]
                    n_tokens_to_unmask = tokens_per_step
                    if "random" in order:
                        masked_indices = masked_indices[torch.randperm(n_masked)]
                    elif "attribute" in order:
                        pass
                    elif "lowest_entropy" in order:
                        masked_probs = flat_probs[masked_indices]
                        entropy = -torch.sum(masked_probs * torch.log(masked_probs), dim=-1)
                        masked_indices = masked_indices[torch.argsort(entropy)]
                    elif "highest_entropy" in order:
                        masked_probs = flat_probs[masked_indices]
                        entropy = -torch.sum(masked_probs * torch.log(masked_probs), dim=-1)
                        masked_indices = masked_indices[torch.argsort(entropy, descending=True)]
                    elif "event" in order:
                        # mod by number of attributes
                        masked_indices_event_index = masked_indices % n_attributes
                        masked_indices = masked_indices[torch.argsort(masked_indices_event_index)]
                    if "reverse" in order:
                        masked_indices = masked_indices.flip(0)
                    indices_to_unmask = masked_indices[:n_tokens_to_unmask]

                    # replace with sampled values
                    flat_x[indices_to_unmask] = torch.nn.functional.one_hot(
                        sampled[indices_to_unmask], num_classes=flat_x.shape[-1]
                    ).to(dtype)
                    # plot

                    x = einops.rearrange(
                        flat_x, "(b ta) v -> b ta v", b=batch, ta=time_attr
                    )

                    updated_event_indices = indices_to_unmask // n_attributes

                    # convert to set
                    updated_event_indices = set(updated_event_indices.cpu().numpy())

                    x = self.tokenizer.collapse_undefined_attributes(x)
                    # x = self.tokenizer.sanitize_mask(x, event_indices=updated_event_indices)


                    # masekd tokens after
                    masked_tokens_after = (x.sum(-1) > 1).sum().int().item()

                    pbar.update(masked_tokens_before - masked_tokens_after)
                    if masked_tokens_after == 0:
                        break
        return x
    
    def step(self, batch, batch_idx):

        x = torch.nn.functional.one_hot(batch["token_ids"], num_classes=len(self.vocab)).float()

        batch_size = x.shape[0]
        ta = x.shape[1]
        v = x.shape[2]

        if self.random_add_masking_type == "mml":
            masked_x = random_add_masking_mml(x)
        elif self.random_add_masking_type == "variable_superposition":
            masked_x = random_add_masking_variable_superposition(x)

        masked_x = masked_x * self.format_mask[None, ...].to(masked_x.device)
        target = x

        logits = self(masked_x)
        target_idx = torch.argmax(target, dim=-1)

        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_idx.reshape(-1),
            reduction = "none",
        )

        prior = masked_x / masked_x.sum(dim=-1, keepdim=True)

        prior_log_probs = torch.log(prior + 1e-8)

        prior_ce = F.cross_entropy(
            prior_log_probs.reshape(-1, prior.shape[-1]),
            target_idx.reshape(-1),
            reduction = "none",
        )

        prior_entropy = -torch.sum(prior * prior_log_probs, dim=-1)
        assert prior_entropy.shape == (batch_size, ta)

        # take mean of entropy
        prior_entropy = prior_entropy.mean(1, keepdim=True)

        ce_reshaped = einops.rearrange(ce, "(b ta) -> b ta", b=batch_size)

        na = ce_reshaped.shape[-1]        
        # now divide by prior entropy
        prior_entropy_scaled_ce = ce_reshaped / (prior_entropy+1)

        known_positions = (masked_x.sum(dim=-1) == 1).flatten()
        ce[known_positions] = 0
        # reshape to batch, loss
        ce = ce.reshape(batch_size, -1)
        metrics = {}
        metrics["cross_entropy"] = ce.mean()
        metrics["prior_entropy_scaled_ce"] = prior_entropy_scaled_ce.mean()


        weights = 1/batch["n_loops_in_parent_song"].float()

        # now scale so sum matches batch size
        weights = weights / weights.sum() * batch_size

        # now calculate cross entropy and prior entropy scaled cross entropy
        metrics["cross_entropy_weighted_by_song"] = (ce / weights).mean()
        metrics["prior_entropy_scaled_ce_weighted_by_song"] = (prior_entropy_scaled_ce / weights).mean()

        with torch.no_grad():
            decoder_output_probs = F.softmax(logits, dim=-1)
            probability = torch.gather(
                decoder_output_probs, dim=-1, index=target_idx.unsqueeze(-1)
            ).squeeze(-1)
            
            # Global metrics
            metrics["probability"] = probability.mean()

            # scale by song
            metrics["probability_weighted_by_song"] = (probability / weights).mean()
            
            # Per attribute metrics
            probability_by_attr = einops.rearrange(
                probability, "b (t a) -> b t a", 
                a=self.n_attributes
            )
            for i, attr in enumerate(self.tokenizer.note_attribute_order):
                metrics[f"probability/{attr}"] = probability_by_attr[:,:,i].mean()
            
            decoder_output_probs_sort = torch.argsort(
                decoder_output_probs, dim=-1, descending=True
            )
   
            # Global accuracy metrics
            for k in [1, 2, 4]:
                accuracy = (
                    target_idx.unsqueeze(-1) == decoder_output_probs_sort[:, :, :k]
                ).any(dim=-1).float()
                
                metrics[f"accuracy@{k}"] = accuracy.mean()

                # scale by song
                metrics[f"accuracy_weighted_by_song@{k}"] = (accuracy / weights).mean()
                
                # Per attribute accuracy
                accuracy_by_attr = einops.rearrange(
                    accuracy, "b (t a) -> b t a",
                    a=self.n_attributes
                )
                for i, attr in enumerate(self.tokenizer.note_attribute_order):
                    metrics[f"accuracy@{k}/{attr}"] = accuracy_by_attr[:,:,i].mean()
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.step(batch, batch_idx)
        for metric in metrics:
            self.log(f"trn/{metric}", metrics[metric], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.use_prior_scaled_ce_loss:
            if self.weight_by_loops_in_parent_song:
                loss = metrics["prior_entropy_scaled_ce_weighted_by_song"]
            else:
                loss = metrics["prior_entropy_scaled_ce"]
        else:
            if self.weight_by_loops_in_parent_song:
                loss = metrics["cross_entropy_weighted_by_song"]
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
        if self.use_prior_scaled_ce_loss:
            loss = metrics["prior_entropy_scaled_ce"]
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

    # BATCH_SIZE = 60 12, 768
    #BATCH_SIZE = 80

    BATCH_SIZE = 80

    tag_list = open(f"./data/{DATASET}/tags.txt").read().splitlines()

    N_BARS = 4 if DATASET == "harmonic" else 4

    tokenizer_config = {
        "ticks_per_beat": 24 if (DATASET == "mmd_loops" or DATASET ==  "harmonic") else 48,
        "time_hierarchy": "tick",
        "pitch_range": [0, 128],
        "max_beats": 4 * N_BARS,
        "max_notes": 75 * N_BARS if DATASET == "mmd_loops" else 20 * N_BARS,
        "min_tempo": 40,
        "max_tempo": 300,
        "n_tempo_bins": 32,
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
        # "use_durations": True,
        # "durations": [Fraction(1, 32), Fraction(1, 16), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(1, 1), Fraction(2, 1), Fraction(4, 1)],
    }

    USE_RANDOM_SHIFT = False
    tokenizer = MergedTokenizer(tokenizer_config)

    print(f"Vocab size: {len(tokenizer.vocab)}")

    # print note attribute order
    print(tokenizer.note_attribute_order)

    model = SuperposedLanguageModel(
        hidden_size=768,
        n_heads=12,
        feed_forward_size=4*768,
        n_layers=12,
        vocab=tokenizer.vocab,
        max_seq_len=tokenizer_config["max_notes"] * len(tokenizer.note_attribute_order),
        learning_rate=1e-4,
        tokenizer_config=tokenizer_config,
        learning_rate_gamma=0.99,
        norm_first=True,
        enforce_constraint_in_forward=True,
        normalize_input=True,
        activation="gelu",
        dropout=0.1,
        use_cross_entropy_increase_loss=False,
        use_prior_scaled_ce_loss=True,
        use_output_bias=False,
        weight_by_loops_in_parent_song=False,
        random_add_masking_type="variable_superposition"
    )

    FINETUNE = False

    if FINETUNE:

        src_model = SuperposedLanguageModel.load_from_checkpoint(
            "./checkpoints/zesty-dawn-376/last.ckpt",
            map_location="cpu",
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

        model.initialize_from_different_model(src_model)

    mmd_4bar_filter_fn = lambda x: f"n_bars={N_BARS}" in x

    # if a track has program 0 and is not a drum track and does not contain the word "piano" in the name, filter out the whole midi
    # we can't risk having mislabelled tracks in the dataset
    sm_filter_fn = lambda sm: not any(
        track.program == 0 and not track.is_drum and "piano" not in track.name.lower()
        for track in sm.tracks
    )

    val_ds = MidiDataset(
        cache_path=f"./data/{DATASET}/val_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn if DATASET == "mmd_loops" else None,
        genre_list=tag_list,
        tokenizer=tokenizer,
        min_notes=4 * N_BARS if DATASET == "mmd_loops" else 4 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
        use_random_shift=USE_RANDOM_SHIFT,
        sm_filter_fn=sm_filter_fn,
    )

    val_tokens = val_ds[0]["token_ids"]

    token_2_count = {t: 0 for t in tokenizer.vocab}
    tokens = []
    # take 10 samples and save tokens
    for i in tqdm(range(1000)):
        val_token_ids = val_ds[i]["token_ids"]
        val_tokens = tokenizer.indices_to_tokens(val_token_ids)
        tokens.append(val_tokens)

    for token in tokens:
        for t in token:
            token_2_count[t] += 1

    # print token counts in token order
    for t in tokenizer.vocab:
        print(f"{t}: {token_2_count[t]}")

    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # take one batch
    # sample = next(iter(val_dl))
    # # try a validation step 
    # model.validation_step(sample, 0)
    print(f"Validation step successful")

    print(f"Loaded {len(val_ds)} validation records")

    trn_ds = MidiDataset(
        cache_path=f"./data/{DATASET}/trn_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn if DATASET == "mmd_loops" else None,
        genre_list=tag_list,
        tokenizer=tokenizer,
        transposition_range=[-6, 6] if DATASET == "mmd_loops" or DATASET == "harmonic" else None,
        min_notes=4 * N_BARS if DATASET == "mmd_loops" else 4 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
        use_random_shift=USE_RANDOM_SHIFT,
        sm_filter_fn=sm_filter_fn,
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
        precision="16-mixed",
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
        # gradient_clip_val=1.0,
        # accumulate_grad_batches=4,
        check_val_every_n_epoch=1 if DATASET == "mmd_loops" else 10,
    )

    trainer.fit(
        model,
        trn_dl,
        val_dl,
    )