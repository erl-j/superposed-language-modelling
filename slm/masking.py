import torch
import einops
import random


def random_superposition(x, syntax_mask):
    '''
    Returns a random superposition of the input tensor x.
    Args:
        x: Input tensor of shape (batch_size, num_events, num_attributes, vocab_size)
        syntax_mask: Mask tensor of shape (num_attributes, vocab_size)
        Syntax mask is a binary tensor where 1 indicates a valid token for that position and 0 indicates an invalid token.
    Returns:
        New tensor of shape (batch_size, num_events, num_attributes, vocab_size) where a random superposition has been applied.  
    '''

    device = x.device
    #  move syntax mask to device
    syntax_mask = syntax_mask.to(device)
    syntax_mask = einops.repeat(syntax_mask, 'a v -> b e a v', b=x.shape[0], e=x.shape[1])
    batch_size, num_events, num_attributes, vocab_size = x.shape
    # convert to bits
    x_bits = x > 0.5
    syntax_mask_bits = syntax_mask > 0.5
    # if in syntax mask and not in x, set to 1
    candidate_confounders = syntax_mask_bits & ~x_bits
    # now create a random mask and set all non confounders to 0

    sup_base = torch.rand_like(x, device=device) * candidate_confounders
    # get max value
    sup_base_max = sup_base.max(dim=-1, keepdim=True).values

    sup_rate = torch.rand(batch_size, num_events, num_attributes, 1, device=device) * (sup_base_max) 
    confounders = sup_base >= sup_rate
    output = torch.clamp(x + confounders, 0, 1)
    # # assert that outputs is between 0 and 1
    # assert (output >= 0).all()
    # # assert than sums are between 1 and len(tokenizer.vocab)
    # assert (output.sum(dim=-1) >= 1).all()
    # assert (output.sum(dim=-1) <= vocab_size).all()
    return output

def position_mask(batch_size,n_positions, device):
    '''
    Returns a random mask of shape (batch_size, n_positions) where between 1 and n_positions are set to 1.
    Args:
        batch_size: Number of samples in the batch
        n_positions: Number of positions in the sequence
    '''
    mask_base = torch.rand(batch_size, n_positions, device=device)
    # get max mask base values
    mask_base_max = mask_base.max(dim=-1, keepdim=True).values
    # now get the mask rate
    mask_rate = torch.rand(batch_size,1, device=device) * mask_base_max
    # get mask
    mask = mask_base >= mask_rate
    return mask

def ratio_superposition(x, syntax_mask, hierarchical_masks = ["attribute", "event", "event_attribute"], superpositions = ["sparse", "full"]):
    """
    Applies superposition masking scheme to a batch of one-hot encoded tensors.

    Args:
        x: Input tensor of shape (batch_size, num_events, num_attributes, vocab_size)
        syntax_mask: Mask tensor of shape (batch_size, num_events, num_attributes, vocab_size)
        Syntax mask is a binary tensor where 1 indicates a valid token for that position and 0 indicates an invalid token.
    Returns:
        Masked tensor of shape (batch_size, num_events, num_attributes, vocab_size)
    """

    device = x.device
    batch_size, num_events, num_attributes, vocab_size = x.shape

    attribute_mask = position_mask(batch_size, num_attributes, device)
    attribute_mask = einops.rearrange(attribute_mask, 'b a -> b 1 a 1')

    event_mask = position_mask(batch_size, num_events, device)
    event_mask = einops.rearrange(event_mask, 'b e -> b e 1 1')

    event_attribute_mask = position_mask(batch_size, num_events * num_attributes, device)
    event_attribute_mask = einops.rearrange(event_attribute_mask, 'b (e a) -> b e a 1', e = num_events, a = num_attributes)

    hierarchical_masks_dict = {"attribute": einops.repeat(attribute_mask, 'b 1 a 1 -> b e a v', v=vocab_size, e=num_events),
                                 "event": einops.repeat(event_mask, 'b e 1 1 -> b e a v', a=num_attributes, v=vocab_size),
                                 "event_attribute": einops.repeat(event_attribute_mask, 'b e a 1 -> b e a v', v=vocab_size)
                            }
    
    syntax_mask = syntax_mask.to(device)
    # get superposition
    superposition = random_superposition(x, syntax_mask)
    full_mask = einops.repeat(syntax_mask, 'a v -> b e a v', b=batch_size, e=num_events)

    superpositions_dict = {"sparse": superposition, "full": full_mask}

    hierarchical_mask_stack = torch.stack([hierarchical_masks_dict[mask_type] for mask_type in hierarchical_masks])
    superposition_stack = torch.stack([superpositions_dict[mask_type] for mask_type in superpositions])

    hierarchical_mask_idx = torch.randint(0, len(hierarchical_masks), (batch_size,), device=device)
    superposition_idx = torch.randint(0, len(superpositions), (batch_size,), device=device)

    hierarchical_mask = hierarchical_mask_stack[hierarchical_mask_idx, torch.arange(batch_size, device=device)]
    superposition = superposition_stack[superposition_idx, torch.arange(batch_size, device=device)]

    masked_x = torch.clamp(hierarchical_mask * superposition + x, 0, 1)

    return masked_x




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

def random_add_masking_variable_superposition(x, superposition_prob_function=lambda x:x):
    batch_size = x.shape[0]
    position_masking_probs = torch.rand(batch_size, device=x.device)
    position_mask = (
        torch.rand((x.shape[0], x.shape[1]), device=x.device)
        < position_masking_probs[:, None]
    )
    # Instead of per-sample superposition probs, use per-position probs
    superposition_probs = superposition_prob_function(torch.rand((x.shape[0], x.shape[1]), device=x.device))
    superposition = (
        torch.rand_like(x, device=x.device) < superposition_probs[:, :, None]
    )
    mask = position_mask[:, :, None] * superposition
    masked_x = torch.clamp(x + mask, 0, 1)
    return masked_x

def mixed_superposition(x):
    """Apply superposition masking scheme to a batch of one-hot encoded tensors.

    Args:
        x: Input tensor of shape (batch_size, num_events, num_attributes, vocab_size)

    Returns:
        Masked tensor of shape (batch_size, num_events, num_attributes, vocab_size)
    """

    device = x.device
    batch_size, num_events, num_attributes, vocab_size = x.shape

    # Sample per-batch parameters
    is_full_mask = (
        torch.rand(batch_size, device=device) < 0.5
    )  # 50% chance of full masking
    masking_type = torch.rand(batch_size, device=device)  # Uniform for type selection

    # Initialize mask for tracking masked positions
    mask = torch.zeros(
        (batch_size, num_events, num_attributes), dtype=torch.bool, device=device
    )

    # Create unmasked positions mask with Bernoulli trials
    known_mask = torch.rand(
        (batch_size, num_events, num_attributes), device=device
    ) < torch.rand((batch_size, 1, 1), device=device)

    # Handle attribute-level masking
    attr_mask_samples = torch.rand(
        (batch_size, 1, num_attributes), device=device
    ) < torch.rand((batch_size, 1, 1), device=device)
    attr_mask = attr_mask_samples.expand(-1, num_events, -1)

    # Handle event-level masking
    event_mask_samples = torch.rand(
        (batch_size, num_events, 1), device=device
    ) < torch.rand((batch_size, 1, 1), device=device)
    event_mask = event_mask_samples.expand(-1, -1, num_attributes)

    # Handle position-level masking
    pos_mask = torch.rand(
        (batch_size, num_events, num_attributes), device=device
    ) < torch.rand((batch_size, 1, 1), device=device)

    # Combine masks based on masking type
    type_is_attr = masking_type[:, None, None] < 1 / 3
    type_is_event = (masking_type[:, None, None] >= 1 / 3) & (
        masking_type[:, None, None] < 2 / 3
    )
    type_is_pos = masking_type[:, None, None] >= 2 / 3

    mask = (
        type_is_attr * attr_mask + type_is_event * event_mask + type_is_pos * pos_mask
    )

    # Apply known_mask to prevent masking those positions
    mask = mask & ~known_mask

    # Create output tensor starting with copy of input
    output = x.clone()

    # Apply full masking where needed
    full_mask_expanded = is_full_mask[:, None, None, None].expand(
        -1, num_events, num_attributes, vocab_size
    )
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, vocab_size)

    # Where we have full masking, set all vocabulary positions to 1
    output = torch.where(
        full_mask_expanded & mask_expanded, torch.ones_like(output), output
    )

    # For sparse masking, do Bernoulli trials for each vocabulary position
    sparse_positions = ~is_full_mask[:, None, None, None] & mask_expanded
    if sparse_positions.any():
        vocab_mask = torch.rand(
            batch_size, num_events, num_attributes, vocab_size, device=device
        ) < torch.rand((batch_size, 1, 1, 1), device=device)
        output = torch.where(
            sparse_positions & vocab_mask, torch.ones_like(output), output
        )

    return output

def mixed_superposition_2(x, mlm=False, 
    hierarchy_mask_types=['attribute', 'event', 'event_attribute'],
    second_mask_types=['full', 'full', 'full', 'shared_superposition', 'variable_superposition', 'variable_superposition_shared_prob'],
    ):
    """Apply superposition masking scheme to a batch of one-hot encoded tensors.

    Args:
        x: Input tensor of shape (batch_size, num_events, num_attributes, vocab_size)
        mlm: If True, returns masked tensor with additional mask channel
        second_mask_types: List of strings specifying which secondary masks to use.
            Valid options: ['full', 'shared_superposition', 'variable_superposition', 
                          'variable_superposition_shared_prob']
            If None, uses all available mask types
        hierarchy_mask_types: List of strings specifying which hierarchy masks to use.
            Valid options: ['attribute', 'event', 'event_attribute']
            If None, uses all available mask types

    Returns:
        If mlm is True, returns masked tensor of shape (batch_size, num_events, num_attributes, vocab_size + 1)
        where the last channel indicates masked positions (1 = masked, 0 = unmasked)
        Else returns masked tensor of shape (batch_size, num_events, num_attributes, vocab_size)
    """
    device = x.device
    batch_size, num_events, num_attributes, vocab_size = x.shape

    # Initialize mask dictionaries
    hierarchy_masks_dict = {}
    second_masks_dict = {}
    
    # Generate hierarchy masks
    hierarchy_mask_prob = torch.rand(batch_size, 1, 1, 1, device=device)
    
    hierarchy_masks_dict['attribute'] = torch.rand(batch_size, 1, num_attributes, 1, device=device) < hierarchy_mask_prob
    hierarchy_masks_dict['event'] = torch.rand(batch_size, num_events, 1, 1, device=device) < hierarchy_mask_prob
    hierarchy_masks_dict['event_attribute'] = torch.rand(batch_size, num_events, num_attributes, 1, device=device) < hierarchy_mask_prob

    # Generate secondary masks
    full_mask = torch.ones_like(x, device=device)
    second_masks_dict['full'] = full_mask

    shared_superposition_probs = torch.rand(batch_size, 1, 1, 1, device=device)
    second_masks_dict['shared_superposition'] = (
        torch.rand(batch_size, 1, 1, vocab_size, device=device) < shared_superposition_probs
    )

    variable_superposition_probs = torch.rand(batch_size, num_events, num_attributes, 1, device=device)
    second_masks_dict['variable_superposition'] = (
        torch.rand_like(x, device=device) < variable_superposition_probs
    )

    variable_superposition_shared_prob_prob = torch.rand(batch_size, 1, 1, 1, device=device)
    second_masks_dict['variable_superposition_shared_prob'] = (
        torch.rand_like(x, device=device) < variable_superposition_shared_prob_prob
    )

    # Use default masks if none specified
    if second_mask_types is None:
        second_mask_types = list(second_masks_dict.keys())
    if hierarchy_mask_types is None:
        hierarchy_mask_types = list(hierarchy_masks_dict.keys())

    # Validate mask types
    for mask_type in second_mask_types:
        if mask_type not in second_masks_dict:
            raise ValueError(f"Invalid second mask type: {mask_type}")
    for mask_type in hierarchy_mask_types:
        if mask_type not in hierarchy_masks_dict:
            raise ValueError(f"Invalid hierarchy mask type: {mask_type}")

    # Stack selected masks
    second_masks = torch.stack([
        second_masks_dict[mask_type].expand_as(x) 
        for mask_type in second_mask_types
    ])
    hierarchy_masks = torch.stack([
        hierarchy_masks_dict[mask_type].expand_as(x) 
        for mask_type in hierarchy_mask_types
    ])

    # Select random masks for each batch
    second_mask_idx = torch.randint(0, len(second_mask_types), (batch_size,), device=device)
    hierarchy_mask_idx = torch.randint(0, len(hierarchy_mask_types), (batch_size,), device=device)

    # Index into stacked masks - each sample gets a different mask
    second_mask = second_masks[second_mask_idx, torch.arange(batch_size, device=device)]
    hierarchy_mask = hierarchy_masks[hierarchy_mask_idx, torch.arange(batch_size, device=device)]

    # Generate position mask
    position_mask_probs = torch.rand(batch_size, 1, 1, 1, device=device)
    position_mask = torch.rand(batch_size, num_events, num_attributes, 1, device=device) < position_mask_probs

    if mlm:
        mask = hierarchy_mask * position_mask
        masked_x = torch.where(mask, torch.zeros_like(x), x)
        masked_x = torch.cat([masked_x, mask], dim=-1)
        return masked_x
    else:
        masked_x = torch.clamp(hierarchy_mask * second_mask * position_mask + x, 0, 1)
        return masked_x

def mlm_mixed(x):
    """
    Applies masking for Masked Language Modeling (MLM) training.

    Args:
        x: Input tensor of shape (batch_size, events, attributes, vocab_size)
           containing one-hot encoded tokens

    Returns:
        Masked tensor of shape (batch_size, events, attributes, vocab_size + 1)
        where the first channel indicates masked positions (1 = masked, 0 = unmasked) if mask_first=True
        otherwise the last channel indicates masked positions.
        and the remaining channels contain the original tokens (zeroed out at masked positions)
    """
    mixed_sup = mixed_superposition(x)

    # find where more than one token is active
    vocab_sums = mixed_sup.sum(dim=-1)

    # create mask based on where more than one token is active
    mask = vocab_sums > 1

    # where mask is True, set to 0, otherwise keep original values
    masked_x = torch.where(mask[:, :, :, None], torch.zeros_like(mixed_superposition), mixed_superposition)

    # add mask channel
    masked_x = torch.cat((mask[:, :, :, None].float(), masked_x), dim=-1)

    return masked_x

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


def attribute_masking(x, masking_prob):
    '''
    Applies attribute masking to input tensor x. For sample and attribute.
    Meaning that 
    with probability masking_prob, that attribute is set to all ones across the entire sequence.
    Args:
        x: Input tensor of shape (batch_size, n_events, n_attributes, vocab_size)
        masking_prob: Probability of masking out an attribute
    '''
    # get dropout mask of shape (batch_size, n_attributes)
    dropout_mask = torch.rand(x.shape[0], x.shape[2], device=x.device) < masking_prob
    # expand the mask to the shape of x
    dropout_mask = einops.rearrange(dropout_mask, 'b a -> b 1 a 1')
    # where dropout mask is True, set x to ones, otherwise keep original values
    x = torch.where(dropout_mask, torch.ones_like(x), x)
    return x

def event_masking(x, masking_prob):
    '''
    Applies event masking to input tensor x. For sample and event.
    Meaning that 
    with probability masking_prob, that event is set to all ones across the entire sequence.
    Args:
        x: Input tensor of shape (batch_size, n_events, n_attributes, vocab_size)
        masking_prob: Probability of masking out an event
    '''
    # get dropout mask of shape (batch_size, n_attributes)
    dropout_mask = torch.rand(x.shape[0], x.shape[1], device=x.device) < masking_prob
    # expand the mask to the shape of x
    dropout_mask = einops.rearrange(dropout_mask, 'b e -> b e 1 1')
    # where dropout mask is True, set x to ones, otherwise keep original values
    x = torch.where(dropout_mask, torch.ones_like(x), x)
    return x
    
    
