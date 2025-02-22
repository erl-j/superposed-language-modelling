from masking import ratio_superposition, random_superposition, mixed_superposition_2, simple_superposition
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Callable, Optional, Tuple

def generate_data(attribute_vocab_sizes: List[int], n_events: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data and syntax mask for testing superposition schemes."""
    # Calculate attribute ranges
    attribute_ranges = []
    last_index = 0
    for size in attribute_vocab_sizes:
        attribute_ranges.append((last_index, last_index + size))
        last_index += size
    
    total_vocab_size = sum(attribute_vocab_sizes)
    
    # Generate sequence of events for each attribute
    x = np.stack(
        [np.random.randint(low=r[0], high=r[1], size=(batch_size, n_events)) 
         for r in attribute_ranges],
        axis=2
    )
    
    # One-hot encode events
    events = np.eye(total_vocab_size)[x]
    
    # Create syntax mask
    syntax_mask = np.zeros((len(attribute_vocab_sizes), total_vocab_size))
    for a_idx, (start, end) in enumerate(attribute_ranges):
        syntax_mask[a_idx, start:end] = 1
    
    return torch.tensor(events, dtype=torch.float), torch.tensor(syntax_mask, dtype=torch.float)

def compare_superposition_schemes(x: torch.Tensor, 
                                syntax_mask: torch.Tensor,
                                schemes: List[dict],
                                plot_type: str = 'heatmap') -> None:
    """
    Compare different superposition schemes side by side.
    
    Args:
        x: Input tensor of shape (batch_size, n_events, n_attributes, vocab_size)
        syntax_mask: Syntax mask tensor
        schemes: List of dictionaries containing superposition parameters
        plot_type: Type of visualization ('heatmap' or 'histogram')
    """
    n_schemes = len(schemes)
    fig, axes = plt.subplots(1, n_schemes, figsize=(5*n_schemes, 4))
    if n_schemes == 1:
        axes = [axes]
    
    vmin, vmax = float('inf'), float('-inf')
    results = []
    
    # First pass: compute all results and find global min/max
    for scheme in schemes:
        if scheme['type'] == 'mixed':
            result = mixed_superposition_2(x)
        elif scheme['type'] == 'simple':
            result = simple_superposition(x, syntax_mask, **scheme['params'])
        elif scheme['type'] == 'ratio':
            result = ratio_superposition(x, syntax_mask, **scheme['params'])
        
        results.append(result)
        if plot_type == 'heatmap':
            vmin = min(vmin, result[0].reshape(-1, result.shape[-1]).T.min())
            vmax = max(vmax, result[0].reshape(-1, result.shape[-1]).T.max())
    
    # Second pass: plot with consistent scales
    for ax, result, scheme in zip(axes, results, schemes):
        if plot_type == 'heatmap':
            im = ax.imshow(result[0].reshape(-1, result.shape[-1]).T, 
                         interpolation='none', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)
        elif plot_type == 'histogram':
            ax.hist((result.sum(-1) > 1).sum(-1).sum(-1).numpy(), 
                   bins=result.shape[2] * result.shape[1])
        
        ax.set_title(f"{scheme['type'].capitalize()} Superposition")
    
    plt.tight_layout()
    return fig, axes

# Example usage
if __name__ == "__main__":
    # Parameters
    ATTRIBUTE_VOCAB_SIZES = [3, 9, 27]
    N_EVENTS = 30
    BATCH_SIZE = 1000
    
    # Generate data
    x, syntax_mask = generate_data(ATTRIBUTE_VOCAB_SIZES, N_EVENTS, BATCH_SIZE)
    
    # Define different superposition schemes to compare
    schemes = [
        {
            'type': 'mixed',
            'params': {}
        },
        {
            'type': 'simple',
            'params': {
                'superpositions': ['full', 'full', 'full', 'sparse', 'shared', 'shared_rate'],
                'schedule_fn': lambda x: x**(1/4),
                'attribute_masking_rate': 0.05
            }
        },
        {
            'type': 'ratio',
            'params': {
                'superpositions': ['full', 'full', 'sparse', 'shared_rate'],
                'schedule_fn': lambda x: x**(1/4),
                'simulate_autoregression': False
            }
        }
    ]
    
    