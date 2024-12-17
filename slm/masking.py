import torch
import einops

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
        torch.rand((x.shape[0], x.shape[1]), device=x.device) < masking_probs[:, None]
    )
    # create masking ratios
    superposition_probs = torch.rand(batch_size, device=x.device)
    superposition = (
        torch.rand_like(x, device=x.device) < superposition_probs[:, None, None]
    )
    mask = position_mask[:, :, None] * superposition
    masked_x = torch.clamp(x + mask, 0, 1)
    return masked_x


def random_add_masking_variable_superposition(x):
    batch_size = x.shape[0]
    position_masking_probs = torch.rand(batch_size, device=x.device)
    position_mask = (
        torch.rand((x.shape[0], x.shape[1]), device=x.device)
        < position_masking_probs[:, None]
    )
    # Instead of per-sample superposition probs, use per-position probs
    superposition_probs = torch.rand((x.shape[0], x.shape[1]), device=x.device)
    superposition = (
        torch.rand_like(x, device=x.device) < superposition_probs[:, :, None]
    )
    mask = position_mask[:, :, None] * superposition
    masked_x = torch.clamp(x + mask, 0, 1)
    return masked_x


# def random_add_masking_variable_superposition_ratio(x, format_mask):
#     batch_size, seq_len, vocab_size = x.shape
#     position_masking_ratios = torch.rand(batch_size, device=x.device)

#     position_mask = torch.zeros(
#         (batch_size, seq_len), dtype=torch.bool, device=x.device
#     )
#     for i in range(batch_size):
#         n_positions = int(seq_len * position_masking_ratios[i])
#         indices = torch.randperm(seq_len, device=x.device)[:n_positions]
#         position_mask[i, indices] = True

#     superposition_ratios = torch.rand((batch_size, seq_len), device=x.device)
#     superposition = torch.zeros_like(x, dtype=torch.bool, device=x.device)

#     for i in range(batch_size):
#         for j in range(seq_len):
#             if position_mask[i, j]:
#                 valid_vocab = torch.where(format_mask[j] == 1)[0]
#                 n_vocab = int(len(valid_vocab) * superposition_ratios[i, j])
#                 vocab_indices = valid_vocab[
#                     torch.randperm(len(valid_vocab), device=x.device)[:n_vocab]
#                 ]
#                 superposition[i, j, vocab_indices] = True

#     masked_x = torch.clamp(x + superposition, 0, 1)
#     return masked_x


def mlm_mask(x: torch.Tensor, mask_first = True) -> torch.Tensor:
    """
    Applies masking for Masked Language Modeling (MLM) training.

    Args:
        x: Input tensor of shape (batch_size, sequence_length, vocab_size)
           containing one-hot encoded tokens

    Returns:
        Masked tensor of shape (batch_size, sequence_length, vocab_size + 1)
        where the first channel indicates masked positions (1 = masked, 0 = unmasked) if mask_first=True
        otherwise the last channel indicates masked positions.
        and the remaining channels contain the original tokens (zeroed out at masked positions)
    """
    batch_size, seq_length, vocab_size = x.shape

    # Generate random masking probability for each sequence in batch
    batch_mask_probs = torch.rand(batch_size, device=x.device)

    # Create masking matrix - True where token should be masked
    # Shape: (batch_size, sequence_length)
    masking_matrix = (
        torch.rand((batch_size, seq_length), device=x.device)
        < batch_mask_probs[:, None]
    )

    # Convert to mask channel by adding feature dimension
    # Shape: (batch_size, sequence_length, 1)
    mask_channel = masking_matrix[:, :, None]

    # Zero out tokens at masked positions
    # Use logical not (~) instead of (1 - mask)
    masked_tokens = x * (~mask_channel)

    # Concatenate mask channel with masked tokens
    # Convert boolean mask to float for concatenation
    if mask_first:
        masked_x = torch.cat((mask_channel.float(), masked_tokens), dim=-1)
    else:
        masked_x = torch.cat((masked_tokens, mask_channel.float()), dim=-1)
    return masked_x
