
import torch
import torch.nn.functional as F
from tokenizer import Tokenizer
from torch import nn
import einops
from tqdm import tqdm
from util import top_k_top_p_filtering

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
        use_mlm,
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
        self.unembedding_layer = nn.Linear(hidden_size, self.vocab_size, bias=False)
        self.enforce_constraint_in_forward = enforce_constraint_in_forward

    def forward(self, x):
        if self.use_mlm:
            return self.mlm_forward(x)
        else:
            return self.slm_forward(x)

    def mlm_forward_from_constraint(self, x):
        """
        Forward pass for Masked Language Model (MLM) with constraint.
        Input:
            x: Input tensor of shape (batch_size, events, attributes, vocab_size)
        """
        # first, if more than 1 non zero token is present
        mask = (x.sum(-1) > 1).float()
        # multiply with x
        x_masked = x * (1 - mask[..., None])
        # add mask channel at the end
        x_masked = torch.cat([x_masked, mask[..., None]], dim=-1)
        # assert that every column sums to one
        assert torch.allclose(x_masked.sum(-1), torch.ones_like(x_masked[..., 0]))
        # pass through transformer
        logits = self(x_masked)
        return logits
    
    def get_device(self):
        return next(self.parameters()).device
    
    @torch.no_grad()
    @torch.inference_mode()
    def conditional_log_likelihood(self, target, constraint):
        """
        Compute conditional likelihood of a sequence given a constraint.
        Input:
            target: Input tensor of shape (batch_size, events, attributes, vocab_size)
            constraint: Input tensor of shape (batch_size, events, attributes, vocab_size)
        """
        self.eval()
        target = target.to(self.get_device())
        constraint = constraint.to(self.get_device())
        with torch.no_grad():
            if self.use_mlm:
                logits = self.mlm_forward_from_constraint(constraint)
            else:
                logits = self.slm_forward(constraint)
        # target logits
        target_logits = (target * logits).sum(dim=-1)

        print(f"Min logits: {logits.min()}, Max logits: {logits.max()}, Mean logits: {logits.mean()}, Median logits: {logits.median()}")

        # print min max mean median
        print(f"Min target logits: {target_logits.min()}, Max target logits: {target_logits.max()}, Mean target logits: {target_logits.mean()}, Median target logits: {target_logits.median()}")

        probs = F.softmax(logits, dim=-1)
        # add epsilon to avoid log(0)

        probs = probs * constraint
        probs = probs / probs.sum(dim=-1, keepdim=True)
        # get log likelihood
        target_probs = (target * probs).sum(dim=-1)
        # assert that target probs sums to one 
        print(target_probs.shape)
        # count zeros in target probs
        n_zeros = torch.sum(target_probs == 0)
        print(f"Number of zeros in target probs: {n_zeros}")
        # mean prob
        print(target_probs.mean())
        log_likelihood = torch.log(target_probs).mean()
        return log_likelihood

    @torch.no_grad()
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
        """
        Generate completions using Superposed Language Model (SLM) or Masked Language Model (MLM).
        Input:
            constraint: Input tensor of shape (batch_size, events, attributes, vocab_size)
        """
        self.eval()
        # normalize mask
        if sampling_steps is None:
            sampling_steps = self.tokenizer.config["max_notes"] * len(
                self.tokenizer.note_attribute_order
            )
        constraint = x
        dtype = self.embedding_layer.weight.dtype
        # move x to device
        x = x.to(self.get_device())
        # convert to model dtype, (fp32, fp16)
        syntax_mask = self.syntax_mask[None, None, ...].to(x.device).to(dtype)
        x = x.to(dtype)
        with torch.no_grad():
            x = x * syntax_mask
            x = self.tokenizer.collapse_undefined_attributes(x)
            x = self.tokenizer.sanitize_mask(
                x, event_indices=range(self.tokenizer.config["max_notes"])
            )
            batch, events, attributes, vocab_size = x.shape
            masked_tokens = (x.sum(-1) > 1).sum().int().item()
            with tqdm(total=masked_tokens) as pbar:
                while True:
                    masked_tokens_before = (x.sum(-1) > 1).sum().int().item()
                    if self.use_mlm:
                        logits = self.mlm_forward_from_constraint(x)
                    else:
                        logits = self(x)
                    flat_logits = einops.rearrange(logits, "b t a v -> (b t a) v")
                    flat_logits = top_k_top_p_filtering(
                        flat_logits, top_k=top_k, top_p=top_p
                    )
                    t = temperature
                    if attribute_temperature is not None:
                        # turn t into 1,1,a tensor
                        attr_t = (
                            torch.ones(self.n_attributes, device=x.device) * temperature
                        )
                        for k, v in attribute_temperature.items():
                            # get attribute index
                            attr_idx = self.tokenizer.note_attribute_order.index(k)
                            attr_t[attr_idx] = v
                        attr_t = einops.repeat(
                            attr_t,
                            "a -> (b e a) 1",
                            e=self.tokenizer.config["max_notes"],
                            b=batch,
                        )
                        t = attr_t
                    flat_probs = F.softmax(flat_logits / t, dim=-1)
                    # print min max mean
                    flat_x = einops.rearrange(x, "b e a v -> (b e a) v")
                    flat_probs = (flat_probs+1e-6) * flat_x
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
                        entropy = -torch.sum(
                            masked_probs * torch.log(masked_probs), dim=-1
                        )
                        masked_indices = masked_indices[torch.argsort(entropy)]
                    elif "highest_entropy" in order:
                        masked_probs = flat_probs[masked_indices]
                        entropy = -torch.sum(
                            masked_probs * torch.log(masked_probs), dim=-1
                        )
                        masked_indices = masked_indices[
                            torch.argsort(entropy, descending=True)
                        ]
                    elif "event" in order:
                        # mod by number of attributes
                        masked_indices_event_index = masked_indices % n_attributes
                        masked_indices = masked_indices[
                            torch.argsort(masked_indices_event_index)
                        ]
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
        # assert that x respects constraints
        return x

    def slm_forward(self, x):
        """
        Forward pass for Superposed Language Model (SLM).
        Input:
            x: Input tensor of shape (batch_size, events, attributes, vocab_size)
        """
        b, e, a, v = x.shape
        syntax_mask = self.syntax_mask[None, None, ...].to(x.device).to(x.dtype)
        syntax_mask = einops.repeat(syntax_mask, "1 1 a v -> b e a v", b=b, e=e)
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
        logits = einops.repeat(
            logits, "b e v -> b e a v", a=len(self.tokenizer.note_attribute_order)
        ).clone()

        zero = torch.zeros_like(x, device=x.device, dtype=x.dtype)

        # apply syntax mask to logits with appropriate tolerance
        logits[syntax_mask.isclose(zero, rtol=1e-5)] = torch.finfo(logits.dtype).min

        # enforce constraint with appropriate tolerances
        if self.enforce_constraint_in_forward:
            logits[x.isclose(zero, rtol=1e-5)] = torch.finfo(logits.dtype).min
        return logits

    def mlm_forward(self, x_masked):
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
        position_mask = x_masked[:, :, :, -1:]  # Shape: b e a 1
        x_not_masked = x_masked[:, :, :, :-1]  # Shape: b e a v

        # Pass through embedding layer and main block
        x = self.embedding_layer(x_masked)  # Shape: b e a h
        z_prior = x.sum(dim=2)  # Shape: b e h
        z_post = self.main_block(z_prior)  # Shape: b e h
        logits = self.unembedding_layer(z_post)  # Shape: b e v

        # Expand transformer output for per-attribute predictions
        logits = einops.repeat(
            logits, "b e v -> b e a v", a=len(self.tokenizer.note_attribute_order)
        ).clone()

        # Create a mask for positions we want to set to negative infinity
        # For unmasked positions (mask=False), we want -inf everywhere except where x_orig=1
        neg_inf_mask = ~position_mask.bool() & ~x_not_masked.bool()

        # apply syntax mask
        syntax_mask = self.syntax_mask[None, None, ...].to(x.device).to(x.dtype)
        syntax_mask = einops.repeat(syntax_mask, "1 1 a v -> b e a v", b=x.shape[0], e=x.shape[1])
        zero = torch.zeros_like(syntax_mask, device=x.device, dtype=x.dtype)

        # apply syntax mask to logits with appropriate tolerance
        logits[syntax_mask.isclose(zero, rtol=1e-5)] = torch.finfo(logits.dtype).min

        # Apply the mask using where
        logits = torch.where(
            neg_inf_mask,
            torch.tensor(
                torch.finfo(logits.dtype).min, device=logits.device, dtype=logits.dtype
            ),
            logits,
        )

        return logits

