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
from tokenizer import Tokenizer
from torch import nn
import einops
from tqdm import tqdm
from util import top_k_top_p_filtering
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
import math
from masking import mlm_mask, random_add_masking_mml, random_add_masking_variable_superposition


class SuperposedLanguageModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        feed_forward_size,
        n_layers,
        tokenizer_config,
        norm_first,
        enforce_constraint_in_forward,
        activation,
        dropout,
        use_mlm
        ):
        super().__init__()
        self.use_mlm = use_mlm
        self.tokenizer = Tokenizer(tokenizer_config)
        self.syntax_mask = torch.Tensor(self.tokenizer.get_syntax_mask())
        self.vocab_size = len(self.tokenizer.vocab)
        self.embedding_layer = nn.Linear(
            self.vocab_size if not use_mlm else self.vocab_size + 1,
            hidden_size,
            bias=False,
        )
        self.main_block = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=feed_forward_size,
                norm_first=norm_first,
                dropout=dropout,
                batch_first=True,
                activation=activation,
            ),
            num_layers=n_layers,
        )
        self.unembedding_layer = nn.Linear(
            hidden_size, self.vocab_size, bias=False
        )
        self.enforce_constraint_in_forward = enforce_constraint_in_forward

    def forward(self,x):
        if self.use_mlm:
            return self.mlm_forward(x)
        else:
            return self.slm_forward(x)
        
    def mlm_forward_w_constraint(self, x):
        '''
        Forward pass for Masked Language Model (MLM) with constraint.
        Input:
            x: Input tensor of shape (batch_size, events, attributes, vocab_size)
        '''
        # first, if more than 1 non zero token is present
        mask = (x.sum(-1) > 1).float()
        # multiply with x
        x_masked = x * (1-mask[..., None])
        # add mask channel at the end
        x_masked = torch.cat([x_masked, mask[..., None]], dim=-1)
        # assert that every column sums to one
        assert torch.allclose(x_masked.sum(-1), torch.ones_like(x_masked[..., 0]))
        # pass through transformer
        logits = self(x_masked)
        return logits

    def generate(self,
        x,
        sampling_steps=None,
        temperature=1,
        top_p=1,
        top_k=0,
        order="random",
        attribute_temperature=None,
        tokens_per_step=1,
    ):
        '''
        Generate completions using Superposed Language Model (SLM) or Masked Language Model (MLM).
        Input:
            constraint: Input tensor of shape (batch_size, events, attributes, vocab_size)
        '''
        self.eval()
        # normalize mask
        if sampling_steps is None:
            sampling_steps = self.tokenizer.config["max_notes"]*len(self.tokenizer.note_attribute_order)
        dtype = self.embedding_layer.weight.dtype
        # convert to model dtype, (fp32, fp16)
        syntax_mask = self.syntax_mask[None, None, ...].to(x.device).to(dtype)
        x = x.to(dtype)
        with torch.no_grad():
            x = x * syntax_mask
            x = self.tokenizer.collapse_undefined_attributes(x)
            x = self.tokenizer.sanitize_mask(x, event_indices=range(self.tokenizer.config["max_notes"]))
            batch, events, attributes, vocab_size = x.shape
            masked_tokens = (x.sum(-1) > 1).sum().int().item()
            with tqdm(total=masked_tokens) as pbar:
                while True:
                    masked_tokens_before = (x.sum(-1) > 1).sum().int().item()
                    if self.use_mlm:
                        logits = self.mlm_forward_w_constraint(x)
                    else:
                        logits = self(x)
                    flat_logits = einops.rearrange(logits, "b t a v -> (b t a) v")
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
                    flat_x = einops.rearrange(x, "b e a v -> (b e a) v")
                    flat_probs = flat_probs * flat_x
                    # renormalize
                    flat_probs = flat_probs / (flat_probs.sum(dim=-1, keepdim=True))
                    sampled = torch.multinomial(flat_probs, 1).squeeze(-1)
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
                        flat_x, "(b e a) v -> b e a v", b=batch, e=events, a=attributes
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
        print(f"output shape {x.shape}")
        return x

    def slm_forward(self,x):
        '''
        Forward pass for Superposed Language Model (SLM).
        Input:
            x: Input tensor of shape (batch_size, events, attributes, vocab_size)
        '''
        b,e,a,v = x.shape
        syntax_mask = self.syntax_mask[None,None,...].to(x.device).to(x.dtype)
        syntax_mask = einops.repeat(syntax_mask, "1 1 a v -> b e a v", b=b,e=e)
        # apply syntax mask
        x = x * syntax_mask
        # renormalize
        x = x / x.sum(dim=-1, keepdim=True)
        # embed attributes
        z_prior = self.embedding_layer(x)
        z_prior = z_prior.sum(dim=2)

        # pass through transformer, get new
        z_post = self.main_block(z_prior)
        # pass through unembedding layer
        logits = self.unembedding_layer(z_post)
        # now we repeat in attribute dimension a times
        logits = einops.repeat(logits, "b e v -> b e a v", a=len(self.tokenizer.note_attribute_order)).clone()
        
        zero = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        one = torch.ones_like(x, device=x.device, dtype=x.dtype)
        
        # apply syntax mask to logits with appropriate tolerance
        logits[syntax_mask.isclose(zero, rtol=1e-5)] = torch.finfo(logits.dtype).min
        
        # enforce constraint with appropriate tolerances
        if self.enforce_constraint_in_forward:
            logits[x.isclose(zero, rtol=1e-5)] = torch.finfo(logits.dtype).min
            logits[x.isclose(one, rtol=1e-5)] = torch.finfo(logits.dtype).max
        return logits
    

    def mlm_forward(self,x_masked):
        """
        Forward pass for Masked Language Model (MLM).
        Args:
            x_masked: Input tensor of shape (batch_size, events, attributes, vocab_size+1)
                    where the last channel indicates masked positions

        Returns:
            Tensor of shape (batch_size, events, attributes, vocab_size) containing model logits,
            with unmasked positions having extreme negative values (-inf) except for the correct token
        """
        # xshape is (batch_size, events, attributes, vocab_size+1)
        
        # Split mask channel and token values
        position_mask = x_masked[:,:,:,-1:]  # Shape: b e a 1
        x_not_masked = x_masked[:,:,:,:-1]   # Shape: b e a v
        
        # Pass through embedding layer and main block
        x = self.embedding_layer(x_masked)  # Shape: b e a h
        z_prior = x.sum(dim=2)  # Shape: b e h
        z_post = self.main_block(z_prior)  # Shape: b e h
        logits = self.unembedding_layer(z_post)  # Shape: b e v

        # Expand transformer output for per-attribute predictions
        logits = einops.repeat(logits, "b e v -> b e a v", a=len(self.tokenizer.note_attribute_order)).clone()
                
        # Create a mask for positions we want to set to negative infinity
        # For unmasked positions (mask=False), we want -inf everywhere except where x_orig=1
        neg_inf_mask = ~position_mask.bool() & ~x_not_masked.bool()
        
        # Apply the mask using where
        logits = torch.where(
            neg_inf_mask,
            torch.tensor(
                torch.finfo(logits.dtype).min,
                device=logits.device, dtype=logits.dtype),
            logits
        )
        
        return logits

    def mlm_generate():
        pass
    

class TrainingWrapper(pl.LightningModule):

    def __init__(self, model_config, learning_rate, learning_rate_gamma, masking_scheme):
        super().__init__()
        self.save_hyperparameters()
        self.model = SuperposedLanguageModel(**model_config)
        self.learning_rate = learning_rate
        self.learning_rate_gamma = learning_rate_gamma
        self.use_mlm = self.model.use_mlm
        self.masking_scheme = masking_scheme
        self.tokenizer = self.model.tokenizer
        self.syntax_mask = torch.Tensor(self.tokenizer.get_syntax_mask())
        
        # assert that masking_scheme is compatible with use_mlm
        if self.use_mlm:
            assert self.masking_scheme == "mlm"
        else:
            assert self.masking_scheme == "variable_superposition" or self.masking_scheme == "mml"
        pass

    def get_model_dtype(self):
        return self.model.embedding_layer.weight.dtype     

    def get_model_device(self):
        return next(self.model.parameters()).device   

    def step(self,batch, batch_idx):
        '''
        Perform a single forward pass and calculate metrics
        Input:
            batch: dictionary containing batch data (token_ids (batch_size, events, attributes), n_loops_in_parent_song)
            batch_idx: index of batch
        '''
        token_ids = batch["token_ids"]
        target_token_ids = batch["token_ids"]
        # one hot encode token_ids with model dtype
        x = torch.nn.functional.one_hot(token_ids, num_classes=len(self.tokenizer.vocab)).to(self.get_model_dtype())
        # apply masking scheme
        if self.use_mlm:
            x_ta= einops.rearrange(x, "b t a v -> b (t a) v")
            x_masked_ta = mlm_mask(x_ta, mask_first=False)
            x_input = einops.rearrange(x_masked_ta, "b (t a) vm -> b t a vm", a=len(self.tokenizer.note_attribute_order))
        else:
            x_ta = einops.rearrange(x, "b t a v -> b (t a) v")
            if self.masking_scheme == "mml":
                x_masked_ta = random_add_masking_mml(x_ta)
            elif self.masking_scheme == "variable_superposition":
                x_masked_ta = random_add_masking_variable_superposition(x_ta)
            x_masked = einops.rearrange(x_masked_ta, "b (t a) v -> b t a v", a=len(self.tokenizer.note_attribute_order))
            x_masked = x_masked * (self.syntax_mask[None, None, ...].to(self.get_model_dtype())).to(self.get_model_device())
            # renormalize
            x_input = x_masked / x_masked.sum(dim=-1, keepdim=True)
        logits = self.model(x_input)
        # get metrics
        metrics = self.compute_metrics(target_token_ids, logits, batch["n_loops_in_parent_song"])
        return metrics
    
    def stage_step(self,batch, batch_idx, stage):
        metrics = self.step(batch, batch_idx)
        # log metrics
        for metric in metrics:
            self.log(f"{stage}/{metric}", metrics[metric], on_step=True, on_epoch=True, prog_bar=True)
        # calculate loss
        loss = metrics["cross_entropy"]
        # plot loss
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def training_step(self,batch, batch_idx):
        return self.stage_step(batch, batch_idx, "trn")
    
    def validation_step(self,batch, batch_idx):
        return self.stage_step(batch, batch_idx, "val")

    def compute_metrics(self, target_token_ids, logits, n_loops_in_parent_song):
        '''
        Compute metrics for a given batch
        Input:
            target_token_ids: target token ids (batch_size, events, attributes)
            logits: model logits (batch_size, events, attributes, vocab_size)
            n_loops_in_parent_song: number of loops in parent song (batch_size)
        Returns:
            Dictionary containing metrics:
            - Cross entropy (global and per attribute)
            - Entropy (global and per attribute) 
            - Probability (global and per attribute)
            - Accuracy @1, @2, @4 (global and per attribute)
        '''
        batch_size, n_events, n_attributes = target_token_ids.shape
        
        # Calculate cross entropy
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_token_ids.reshape(-1),
            reduction="none",
        )
        ce = ce.reshape(batch_size, n_events * n_attributes)
        
        # Calculate probabilities and entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * (log_probs + 1e-8)).sum(dim=-1)  # (batch_size, events, attributes)
        
        # Get probability of correct tokens
        target_probs = torch.gather(
            probs.reshape(-1, logits.shape[-1]),
            dim=-1,
            index=target_token_ids.reshape(-1).unsqueeze(-1)
        ).reshape(batch_size, n_events, n_attributes)
        
        # Initialize metrics dictionary
        metrics = {
            "cross_entropy": ce.mean(),
            "entropy": entropy.mean(),
            "probability": target_probs.mean(),
        }
        
        # Calculate accuracies for k=[1,2,4]
        topk_values = [1, 2, 4]
        for k in topk_values:
            decoder_output_probs_sort = torch.argsort(probs, dim=-1, descending=True)  # (batch_size, events, attributes, vocab_size)
            accuracy = (
                (target_token_ids.unsqueeze(-1) == decoder_output_probs_sort[..., :k])  # (batch_size, events, attributes, k)
                .any(dim=-1)  # (batch_size, events, attributes)
                .float()
            )
            
            metrics[f"accuracy@{k}"] = accuracy.mean()
            
            # Per-attribute accuracy metrics
            for i, attr in enumerate(self.tokenizer.note_attribute_order):
                metrics[f"accuracy@{k}/{attr}"] = accuracy[:, :, i].mean()
        
        # Per-attribute metrics
        for attr_idx, attribute in enumerate(self.tokenizer.note_attribute_order):
            # Per-attribute entropy
            metrics[f"entropy/{attribute}"] = entropy[:, :, attr_idx].mean()
            
            # Per-attribute probability
            metrics[f"probability/{attribute}"] = target_probs[:, :, attr_idx].mean()
        
        return metrics


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.learning_rate_gamma, step_size=1)
        return [optimizer], [scheduler]
    
    def generate(self, x, sampling_steps=None, temperature=1, top_p=1, top_k=0, order="random", attribute_temperature=None, tokens_per_step=1):
        return self.model.generate(x, sampling_steps, temperature, top_p, top_k, order, attribute_temperature, tokens_per_step)

        


# class SuperposedLanguageModel(pl.LightningModule):
#     def __init__(
#         self,
#         hidden_size,
#         n_heads,
#         feed_forward_size,
#         n_layers,
#         vocab,
#         max_seq_len,
#         learning_rate,
#         tokenizer_config,
#         learning_rate_gamma=0.9,
#         normalize_input=False,
#         norm_first=False,
#         enforce_constraint_in_forward = True,
#         token_atoms = [],
#         use_composite_unembedding = False,
#         pos_embedding_attributes = [],
#         base_period = None,
#         activation = "relu",
#         use_cross_entropy_increase_loss = False,
#         use_prior_scaled_ce_loss = False,
#         dropout = 0.1,
#         use_input_bias = False,
#         use_output_bias = True,
#         use_embedding_l2_normalization = False,
#         use_unembedding_l2_normalization = False,
#         use_mlm = False,
#         weight_by_loops_in_parent_song = False,
#         random_add_masking_type = "mml"
#     ):
#         """
#         seq_len: length of chart sequence (equal or longer to audio sequence)
#         """
#         super().__init__()
#         self.save_hyperparameters()
#         vocab_size = len(vocab)
#         self.tokenizer = Tokenizer(tokenizer_config)
#         self.format_mask = torch.Tensor(self.tokenizer.get_format_mask())
#         self.vocab = vocab
#         self.positional_encoding  = nn.Parameter(torch.zeros(1, max_seq_len*2, hidden_size), requires_grad=True)
#         self.embedding_layer = nn.Linear(vocab_size if not use_mlm else vocab_size + 1, hidden_size, bias=use_input_bias)
     
#         self.transformer = torch.nn.TransformerEncoder(
#             encoder_layer=torch.nn.TransformerEncoderLayer(
#             d_model=hidden_size,
#             nhead=n_heads,
#             dim_feedforward=feed_forward_size,
#             norm_first= norm_first,
#             dropout=dropout,
#             batch_first=True,
#             activation=activation,
#             ),
#             num_layers=n_layers,
#         )
#         self.decoder_output_layer = nn.Linear(hidden_size, vocab_size, bias=use_output_bias)
#         self.seq_len = max_seq_len
#         self.n_attributes = len(self.tokenizer.note_attribute_order)
#         self.learning_rate_gamma = learning_rate_gamma
#         self.enforce_constraint_in_forward = enforce_constraint_in_forward
#         self.normalize_input = normalize_input
#         self.use_cross_entropy_increase_loss = use_cross_entropy_increase_loss
#         self.use_prior_scaled_ce_loss = use_prior_scaled_ce_loss
#         self.weight_by_loops_in_parent_song = weight_by_loops_in_parent_song
#         self.random_add_masking_type = random_add_masking_type
#         self.use_mlm = use_mlm
        

#     def convert_to_half(self):
#         return self.half()
    
#     def mlm_forward(self, x_masked):
#         """
#         Forward pass for Masked Language Modeling (MLM).
        
#         Args:
#             x_masked: Input tensor of shape (batch_size, time*attributes, vocab_size+1)
#                     where the first channel indicates masked positions
        
#         Returns:
#             Tensor of shape (batch_size, time*attributes, vocab_size) containing model logits,
#             with unmasked positions having extreme negative values (-inf) except for the correct token
#         """
#         # Reshape to expose attribute dimension
#         x = einops.rearrange(x_masked, "b (t a) vm -> b t a vm", a=self.n_attributes)
        
#         # Split mask channel and token values
#         position_mask = x[:,:,:,:1]  # Shape: b t a 1
#         x_not_masked = x[:,:,:,1:]   # Shape: b t a v
        
#         # Pass through embedding layer and transformer
#         x = self.embedding_layer(x)  # Shape: b t a h
#         ze = x.sum(dim=2)  # Shape: b t h
#         zl = self.transformer(ze)  # Shape: b t h
        
#         # Expand transformer output for per-attribute predictions
#         note_z = einops.rearrange(zl, "b t ft -> b t 1 ft")  # Shape: b t 1 h
#         note_z = note_z.repeat(1, 1, self.n_attributes, 1)   # Shape: b t a h
        
#         # Generate logits
#         logits = self.decoder_output_layer(note_z)  # Shape: b t a v
        
#         # Reshape everything to sequence form
#         logits = einops.rearrange(logits, "b t a v -> b (t a) v")
#         x_orig = einops.rearrange(x_not_masked, "b t a v -> b (t a) v")
#         mask = einops.rearrange(position_mask.bool().squeeze(-1), "b t a -> b (t a)")
        
#         # Create a mask for positions we want to set to negative infinity
#         # For unmasked positions (mask=False), we want -inf everywhere except where x_orig=1
#         neg_inf_mask = ~mask.unsqueeze(-1) & ~x_orig.bool()
        
#         # Apply the mask using where
#         logits = torch.where(
#             neg_inf_mask,
#             torch.tensor(
#                 torch.finfo(logits.dtype).min,
#                 device=logits.device, dtype=logits.dtype),
#             logits
#         )
        
#         return logits

#     def forward(self, x):
#         if self.use_mlm:
#             return self.mlm_forward(x)
#         format_mask = self.format_mask[None, ...].to(x.device).to(x.dtype)
#         x = x * format_mask

#         x = einops.rearrange(x, "b (t a) v -> b t a v", a=self.n_attributes)
#         x = x.to(self.device)
#         if self.normalize_input:
#             x2 = x / x.sum(dim=-1, keepdim=True)
#         else:
#             x2 = x
#         ze = self.embedding_layer(x2)
#         ze = ze.sum(dim=2)
#         zl = self.transformer(ze)
#         note_z = einops.rearrange(zl, "b t ft -> b t 1 ft")
#         note_z = note_z.repeat(1, 1, self.n_attributes, 1)
#         decoder_logits = self.decoder_output_layer(note_z)
#         if self.enforce_constraint_in_forward:
   
#             # CHANGED THIS TO torch.finfo(decoder_logits.dtype).min FROM torch.finfo(x.dtype).min
#             decoder_logits[x==0] = torch.finfo(decoder_logits.dtype).min
#             decoder_logits = einops.rearrange(decoder_logits, "b t a v -> b (t a) v", a=self.n_attributes)
#             # CHANGED THIS TO torch.finfo(decoder_logits.dtype).min FROM torch.finfo(x.dtype).min
#             decoder_logits[
#                 (format_mask * torch.ones_like(decoder_logits, device=self.device))==0
#             ] = torch.finfo(decoder_logits.dtype).min
#         # crop to decoder length
#         return decoder_logits
    
#     @torch.inference_mode()
#     def fast_kill_events(self, x, temperature=1):
#         self.eval()
#         dtype = self.embedding_layer.weight.dtype
#         x = x.to(dtype)

#         with torch.no_grad():
#             x = x
#             x = x * self.format_mask[None, ...].to(x.device).to(dtype)
#             x = self.tokenizer.collapse_undefined_attributes(x)
#             x = self.tokenizer.sanitize_mask(x, event_indices=range(self.tokenizer.config["max_notes"]))
#             logits = self(x)
#             logits = logits / temperature
#             new_x = self.tokenizer.get_undefined_probs(x, logits)
#         return new_x
    
#     @torch.inference_mode()
#     def mlm_generate(
#         self,
#         x,
#         sampling_steps=None,
#         temperature=1,
#         top_p=1,
#         top_k=0,
#         order="random",
#         attribute_temperature=None,
#         tokens_per_step=1,
#     ):
#         """
#         Generate completions using masked language modeling.
        
#         Args:
#             x: Input tensor (batch_size, time*attributes, vocab_size)
#             sampling_steps: Number of steps to sample for (optional)
#             temperature: Sampling temperature
#             top_p: Nucleus sampling threshold
#             top_k: Top-k sampling threshold
#             order: Token generation order ("random", "attribute", "lowest_entropy", "highest_entropy", "event")
#             attribute_temperature: Dict of temperatures per attribute (optional)
#             tokens_per_step: Number of tokens to generate per step
        
#         Returns:
#             Generated tensor with same shape as input
#         """
#         self.eval()
#         dtype = self.embedding_layer.weight.dtype
#         x = x.to(dtype)
        
#         with torch.no_grad():
#             # Apply format mask and prepare input
#             x = x * self.format_mask[None, ...].to(x.device).to(dtype)
#             x = self.tokenizer.collapse_undefined_attributes(x)
#             x = self.tokenizer.sanitize_mask(x, event_indices=range(self.tokenizer.config["max_notes"]))
#             constraint = x
#             batch, time_attr, vocab_size = x.shape
            
#             # Create mask channel and prepare masked input
#             mask_channel = (x.sum(-1) > 1)[..., None].float()
#             x_part = x.clone()  # Clone to avoid modifying original
#             x_part = x_part * (x_part.sum(-1, keepdim=True) <= 1).float()
#             x_masked = torch.cat([mask_channel, x_part], dim=-1)

#             # Verify initial state
#             assert (x.sum(-1) > 1).sum().int().item() >= 0, "No tokens to generate"
#             masked_tokens = mask_channel.sum().int().item()
            
#             with tqdm(total=masked_tokens) as pbar:
#                 while True:
#                     masked_tokens_before = mask_channel.sum().int().item()
                    
#                     # Get model predictions
#                     logits = self.mlm_forward(x_masked)
#                     flat_logits = einops.rearrange(logits, "b ta v -> (b ta) v")
                    
#                     # Apply sampling modifications
#                     if top_k > 0 or top_p < 1:
#                         flat_logits = top_k_top_p_filtering(flat_logits, top_k=top_k, top_p=top_p)
                    
#                     # Handle attribute-specific temperatures
#                     t = temperature
#                     if attribute_temperature is not None:
#                         attr_t = torch.ones(self.n_attributes, device=x.device) * temperature
#                         for k, v in attribute_temperature.items():
#                             attr_idx = self.tokenizer.note_attribute_order.index(k)
#                             attr_t[attr_idx] = v
#                         attr_t = einops.repeat(
#                             attr_t, "a -> (b e a) 1",
#                             e=self.tokenizer.config["max_notes"],
#                             b=batch
#                         )
#                         t = attr_t
                    
#                     # Convert to probabilities and apply constraint
#                     flat_probs = F.softmax(flat_logits / t, dim=-1)
#                     flat_probs = flat_probs * constraint.view(-1, vocab_size)
#                     flat_probs = flat_probs / (flat_probs.sum(dim=-1, keepdim=True))
                    
#                     # Sample new tokens
#                     sampled = torch.multinomial(flat_probs, 1).squeeze(-1)
                    
#                     # Find masked positions
#                     flat_mask = einops.rearrange(mask_channel, "b ta 1 -> (b ta)")
#                     masked_indices = torch.where(flat_mask > 0)[0]
#                     n_attributes = len(self.tokenizer.note_attribute_order)
#                     n_masked = masked_indices.shape[0]
                    
#                     if n_masked == 0:
#                         break
                    
#                     # Order masked indices according to strategy
#                     if "random" in order:
#                         masked_indices = masked_indices[torch.randperm(n_masked)]
#                     elif "attribute" in order:
#                         pass
#                     elif "lowest_entropy" in order:
#                         masked_probs = flat_probs[masked_indices]
#                         entropy = -torch.sum(masked_probs * torch.log(masked_probs + 1e-10), dim=-1)
#                         masked_indices = masked_indices[torch.argsort(entropy)]
#                     elif "highest_entropy" in order:
#                         masked_probs = flat_probs[masked_indices]
#                         entropy = -torch.sum(masked_probs * torch.log(masked_probs + 1e-10), dim=-1)
#                         masked_indices = masked_indices[torch.argsort(entropy, descending=True)]
#                     elif "event" in order:
#                         masked_indices_event_index = masked_indices % n_attributes
#                         masked_indices = masked_indices[torch.argsort(masked_indices_event_index)]
                    
#                     if "reverse" in order:
#                         masked_indices = masked_indices.flip(0)
                    
#                     # Select indices to unmask
#                     indices_to_unmask = masked_indices[:tokens_per_step]
                    
#                     # Update tokens
#                     flat_x = einops.rearrange(x, "b ta v -> (b ta) v")
#                     flat_x[indices_to_unmask] = torch.nn.functional.one_hot(
#                         sampled[indices_to_unmask],
#                         num_classes=vocab_size
#                     ).to(dtype)
#                     x = einops.rearrange(flat_x, "(b ta) v -> b ta v", b=batch, ta=time_attr)
                    
#                     # Update mask
#                     flat_mask = flat_mask.clone()
#                     flat_mask[indices_to_unmask] = 0
#                     mask_channel = einops.rearrange(flat_mask, "(b ta) -> b ta 1", b=batch)
                    
#                     # Update x_masked for next iteration
#                     x = self.tokenizer.collapse_undefined_attributes(x)
#                     x_part = x.clone()
#                     x_part = x_part * (x_part.sum(-1, keepdim=True) <= 1).float()
#                     x_masked = torch.cat([mask_channel, x_part], dim=-1)
                    
#                     # Update progress
#                     masked_tokens_after = mask_channel.sum().int().item()
#                     pbar.update(masked_tokens_before - masked_tokens_after)
                    
#                     if masked_tokens_after == 0:
#                         break
            
#             # Verify output
#             assert x.shape == (batch, time_attr, vocab_size), "Output shape mismatch"
#             assert (x.sum(-1) > 1).sum().int().item() == 0, "Invalid token distributions"
#             assert (x < 0).sum().int().item() == 0, "Negative values in output"
#             assert torch.all(x == x * constraint), "Constraint violation"
            
#             return x
        
#     @torch.inference_mode()
#     def generate(
#         self,
#         x,
#         sampling_steps=None,
#         temperature=1,
#         top_p=1,
#         top_k=0,
#         order="random",
#         attribute_temperature=None,
#         tokens_per_step=1,
#     ):
#         if self.use_mlm:
#             return self.mlm_generate(
#                 x,
#                 sampling_steps=sampling_steps,
#                 temperature=temperature,
#                 top_p=top_p,
#                 top_k=top_k,
#                 order=order,
#                 attribute_temperature=attribute_temperature,
#                 tokens_per_step=tokens_per_step,
#             )
#         if sampling_steps is None:
#             sampling_steps = self.tokenizer.config["max_notes"]*len(self.tokenizer.note_attribute_order)
#         self.eval()
#         dtype = self.embedding_layer.weight.dtype
#         # convert to model dtype, (fp32, fp16)
#         x = x.to(dtype)
#         with torch.no_grad():
#             x = x
#             # multiply by format mask
#             x = x * self.format_mask[None, ...].to(x.device).to(dtype)
#             # x = self.tokenizer.normalize_constraint(x
#             x = self.tokenizer.collapse_undefined_attributes(x)
#             # sanitize all events
#             x = self.tokenizer.sanitize_mask(x, event_indices=range(self.tokenizer.config["max_notes"]))
#             batch, time_attr, ft = x.shape
#             # count number of known tokens
#             masked_tokens = (x.sum(-1) > 1).sum().int().item()
#             with tqdm(total=masked_tokens) as pbar:
#                 while True:
#                     masked_tokens_before = (x.sum(-1) > 1).sum().int().item()
#                     # take time of forward pass
#                     logits = self(x)
#                     # invert probs
#                     # flatten
#                     flat_logits = einops.rearrange(logits, "b ta v -> (b ta) v")
#                     flat_logits = top_k_top_p_filtering(flat_logits, top_k=top_k, top_p=top_p)
#                     t = temperature
#                     if attribute_temperature is not None:
#                         # turn t into 1,1,a tensor
#                         attr_t = torch.ones(self.n_attributes, device=x.device) * temperature
#                         for k, v in attribute_temperature.items():
#                             # get attribute index
#                             attr_idx = self.tokenizer.note_attribute_order.index(k)
#                             attr_t[attr_idx] = v
#                         attr_t = einops.repeat(attr_t, "a -> (b e a) 1", e=self.tokenizer.config["max_notes"], b=batch)
#                         t = attr_t
#                     flat_probs = F.softmax(flat_logits / t, dim=-1)
#                     flat_x = einops.rearrange(x, "b ta v -> (b ta) v")
#                     # renormalize
#                     flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
#                     sampled = torch.multinomial(flat_probs, 1).squeeze(-1)
#                     flat_x = einops.rearrange(x, "b ta v -> (b ta) v")
#                     masked_indices = torch.where(flat_x.sum(-1) > 1)[0]
#                     n_attributes = len(self.tokenizer.note_attribute_order)
#                     n_masked = masked_indices.shape[0]
#                     n_tokens_to_unmask = tokens_per_step
#                     if "random" in order:
#                         masked_indices = masked_indices[torch.randperm(n_masked)]
#                     elif "attribute" in order:
#                         pass
#                     elif "lowest_entropy" in order:
#                         masked_probs = flat_probs[masked_indices]
#                         entropy = -torch.sum(masked_probs * torch.log(masked_probs), dim=-1)
#                         masked_indices = masked_indices[torch.argsort(entropy)]
#                     elif "highest_entropy" in order:
#                         masked_probs = flat_probs[masked_indices]
#                         entropy = -torch.sum(masked_probs * torch.log(masked_probs), dim=-1)
#                         masked_indices = masked_indices[torch.argsort(entropy, descending=True)]
#                     elif "event" in order:
#                         # mod by number of attributes
#                         masked_indices_event_index = masked_indices % n_attributes
#                         masked_indices = masked_indices[torch.argsort(masked_indices_event_index)]
#                     if "reverse" in order:
#                         masked_indices = masked_indices.flip(0)
#                     indices_to_unmask = masked_indices[:n_tokens_to_unmask]

#                     # replace with sampled values
#                     flat_x[indices_to_unmask] = torch.nn.functional.one_hot(
#                         sampled[indices_to_unmask], num_classes=flat_x.shape[-1]
#                     ).to(dtype)
#                     # plot

#                     x = einops.rearrange(
#                         flat_x, "(b ta) v -> b ta v", b=batch, ta=time_attr
#                     )

#                     updated_event_indices = indices_to_unmask // n_attributes

#                     # convert to set
#                     updated_event_indices = set(updated_event_indices.cpu().numpy())

#                     x = self.tokenizer.collapse_undefined_attributes(x)
#                     # x = self.tokenizer.sanitize_mask(x, event_indices=updated_event_indices)


#                     # masekd tokens after
#                     masked_tokens_after = (x.sum(-1) > 1).sum().int().item()

#                     pbar.update(masked_tokens_before - masked_tokens_after)
#                     if masked_tokens_after == 0:
#                         break
#         return x
    
#     def step(self, batch, batch_idx):
#         x = torch.nn.functional.one_hot(
#             batch["token_ids"], num_classes=len(self.vocab)
#         ).float()

#         batch_size = x.shape[0]
#         ta = x.shape[1]
#         v = x.shape[2]

#         if self.use_mlm:
#             masked_x = mlm_mask(x)
#         else:
#             if self.random_add_masking_type == "mml":
#                 masked_x = random_add_masking_mml(x)
#             elif self.random_add_masking_type == "variable_superposition":
#                 masked_x = random_add_masking_variable_superposition(x)
    
#             masked_x = masked_x * self.format_mask[None, ...].to(masked_x.device)
        
#         target = x

#         logits = self(masked_x)
#         target_idx = torch.argmax(target, dim=-1)

#         ce = F.cross_entropy(
#             logits.reshape(-1, logits.shape[-1]),
#             target_idx.reshape(-1),
#             reduction="none",
#         )

#         # prior = masked_x / masked_x.sum(dim=-1, keepdim=True)
#         # prior_log_probs = torch.log(prior + 1e-8)
#         # prior_ce = F.cross_entropy(
#         #     prior_log_probs.reshape(-1, prior.shape[-1]),
#         #     target_idx.reshape(-1),
#         #     reduction="none",
#         # )
#         # prior_entropy = -torch.sum(prior * prior_log_probs, dim=-1)
#         # assert prior_entropy.shape == (batch_size, ta)
#         # # take mean of entropy
#         # prior_entropy = prior_entropy.mean(1, keepdim=True)

#         # ce_reshaped = einops.rearrange(ce, "(b ta) -> b ta", b=batch_size)
#         # # now divide by prior entropy
#         # prior_entropy_scaled_ce = ce_reshaped / (prior_entropy + 1)

#         # known_positions = (masked_x.sum(dim=-1) == 1).flatten()
#         # ce[known_positions] = 0
#         # reshape to batch, loss
#         ce = ce.reshape(batch_size, -1)
#         metrics = {}
#         metrics["cross_entropy"] = ce.mean()
#         # metrics["prior_entropy_scaled_ce"] = prior_entropy_scaled_ce.mean()

#         weights = 1 / batch["n_loops_in_parent_song"].float()
#         weights = weights / weights.sum() * batch_size

#         metrics["cross_entropy_weighted_by_song"] = (ce / weights).mean()
#         # metrics["prior_entropy_scaled_ce_weighted_by_song"] = (
#         #     prior_entropy_scaled_ce / weights
#         # ).mean()

#         with torch.no_grad():
#             decoder_output_probs = F.softmax(logits, dim=-1)

#             # Calculate entropy of predictions
#             log_probs = F.log_softmax(logits, dim=-1)
#             entropy = -(decoder_output_probs * log_probs).sum(
#                 dim=-1
#             )  # Shape: [batch_size, time*attributes]

#             # Log global entropy
#             metrics["entropy"] = entropy.mean()

#             # Calculate and log per-attribute entropy
#             entropy_by_attr = einops.rearrange(
#                 entropy, "b (t a) -> b t a", a=self.n_attributes
#             )
#             for i, attr in enumerate(self.tokenizer.note_attribute_order):
#                 metrics[f"entropy/{attr}"] = entropy_by_attr[:, :, i].mean()

#             probability = torch.gather(
#                 decoder_output_probs, dim=-1, index=target_idx.unsqueeze(-1)
#             ).squeeze(-1)

#             # Global metrics
#             metrics["probability"] = probability.mean()
#             metrics["probability_weighted_by_song"] = (probability / weights).mean()

#             # Per attribute metrics
#             probability_by_attr = einops.rearrange(
#                 probability, "b (t a) -> b t a", a=self.n_attributes
#             )
#             for i, attr in enumerate(self.tokenizer.note_attribute_order):
#                 metrics[f"probability/{attr}"] = probability_by_attr[:, :, i].mean()

#             decoder_output_probs_sort = torch.argsort(
#                 decoder_output_probs, dim=-1, descending=True
#             )

#             # Global accuracy metrics
#             for k in [1, 2, 4]:
#                 accuracy = (
#                     (target_idx.unsqueeze(-1) == decoder_output_probs_sort[:, :, :k])
#                     .any(dim=-1)
#                     .float()
#                 )

#                 metrics[f"accuracy@{k}"] = accuracy.mean()
#                 metrics[f"accuracy_weighted_by_song@{k}"] = (accuracy / weights).mean()

#                 # Per attribute accuracy
#                 accuracy_by_attr = einops.rearrange(
#                     accuracy, "b (t a) -> b t a", a=self.n_attributes
#                 )
#                 for i, attr in enumerate(self.tokenizer.note_attribute_order):
#                     metrics[f"accuracy@{k}/{attr}"] = accuracy_by_attr[:, :, i].mean()
#         return metrics

#     def training_step(self, batch, batch_idx):
#         metrics = self.step(batch, batch_idx)
#         for metric in metrics:
#             self.log(f"trn/{metric}", metrics[metric], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
#         if self.use_prior_scaled_ce_loss:
#             if self.weight_by_loops_in_parent_song:
#                 loss = metrics["prior_entropy_scaled_ce_weighted_by_song"]
#             else:
#                 loss = metrics["prior_entropy_scaled_ce"]
#         else:
#             if self.weight_by_loops_in_parent_song:
#                 loss = metrics["cross_entropy_weighted_by_song"]
#             else:
#                 loss = metrics["cross_entropy"]
#         self.log("trn/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
#         # log wandb name
#         self.log("gpu", loss.device.index)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         with torch.no_grad():
#             metrics = self.step(batch, batch_idx)
#         for metric in metrics:
#             self.log(f"val/{metric}", metrics[metric], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
#         if self.use_prior_scaled_ce_loss:
#             loss = metrics["prior_entropy_scaled_ce"]
#         else:
#             loss = metrics["cross_entropy"]
#         self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
#         return loss

#     def configure_optimizers(self):
#         # learning rate decay
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=self.learning_rate_gamma, step_size=1)
#         return [optimizer], [scheduler]

#     def initialize_from_different_model(self, src_model, skip_tokens=[]):
#         for token in self.vocab:
#             if token not in src_model.vocab:
#                 print(f"Token {token} not found in source model")
#                 continue
#             elif any([token.split(":")[0] in skip for skip in skip_tokens]):
#                 print(f"Skipping token {token}")
#                 continue
#             else:
#                 print(f"Transplanting token {token}")
#                 src_idx = src_model.vocab.index(token)
#                 tgt_idx = self.vocab.index(token)
#                 self.embedding_layer.weight.data[:, tgt_idx] = (
#                     src_model.embedding_layer.weight.data[:, src_idx]
#                 )
#                 self.decoder_output_layer.weight.data[tgt_idx, :] = (
#                     src_model.decoder_output_layer.weight.data[src_idx, :]
#                 )
#         # now copy transformer
#         self.transformer.load_state_dict(src_model.transformer.state_dict())

if __name__ == "__main__":

    SEED = 0

    torch.manual_seed(SEED)

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
        'fold_event_attributes' : False,
    }

    USE_RANDOM_SHIFT = False
    tokenizer = Tokenizer(tokenizer_config)

    # model_config = {
    #     "hidden_size": 768,
    #     "n_heads":12,
    #     "feed_forward_size": 4*768,
    #     "n_layers": 12,
    #     "tokenizer_config": tokenizer_config,
    #     "norm_first": True,
    #     "enforce_constraint_in_forward": True,
    #     "activation":"gelu",
    #     "dropout":0.1,
    #     "use_mlm":False
    # }

    # training_wrapper = TrainingWrapper(
    #     model_config=model_config,
    #     learning_rate=1e-4,
    #     learning_rate_gamma=0.99,
    #     masking_scheme="variable_superposition",
    # )

    model_config = {
        "hidden_size": 768,
        "n_heads":12,
        "feed_forward_size": 4*768,
        "n_layers": 12,
        "tokenizer_config": tokenizer_config,
        "norm_first": True,
        "enforce_constraint_in_forward": True,
        "activation":"gelu",
        "dropout":0.1,
        "use_mlm":True
    }

    training_wrapper = TrainingWrapper(
        model_config=model_config,
        learning_rate=1e-4,
        learning_rate_gamma=0.99,
        masking_scheme="mlm",
    )

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

    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

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
        devices=[2,3], 
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
        gradient_clip_val=1.0,
        # accumulate_grad_batches=4,
        check_val_every_n_epoch=1 if DATASET == "mmd_loops" else 10,
    )

    trainer.fit(
        training_wrapper,
        trn_dl,
        val_dl,
        ckpt_path = "./checkpoints/misunderstood-monkey-520/last.ckpt"
        # ckpt_path = "./checkpoints/effortless-sound-516/last.ckpt"
    )