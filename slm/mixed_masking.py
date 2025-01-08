import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def mixed_superposition(x):
    """Apply superposition masking scheme to a batch of one-hot encoded tensors.

    Args:
        x: Input tensor of shape (batch_size, num_events, num_attributes, vocab_size)

    Returns:
        Tuple of (masked_tensor, mask) where mask shows which positions were masked
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

def mixed(x):
    """Apply superposition masking scheme to a batch of one-hot encoded tensors."""
    device = x.device
    batch_size, num_events, num_attributes, vocab_size = x.shape

    # Sample per-batch parameters
    is_full_mask = torch.rand(batch_size, device=device) < 0.5
    masking_type = torch.rand(batch_size, device=device)

    # Initialize mask
    mask = torch.zeros(
        (batch_size, num_events, num_attributes), dtype=torch.bool, device=device
    )

    # Create unmasked positions mask
    known_mask = torch.rand(
        (batch_size, num_events, num_attributes), device=device
    ) < torch.rand((batch_size, 1, 1), device=device)

    # Handle different mask types
    attr_mask_samples = torch.rand(
        (batch_size, 1, num_attributes), device=device
    ) < torch.rand((batch_size, 1, 1), device=device)
    attr_mask = attr_mask_samples.expand(-1, num_events, -1)

    event_mask_samples = torch.rand(
        (batch_size, num_events, 1), device=device
    ) < torch.rand((batch_size, 1, 1), device=device)
    event_mask = event_mask_samples.expand(-1, -1, num_attributes)

    pos_mask = torch.rand(
        (batch_size, num_events, num_attributes), device=device
    ) < torch.rand((batch_size, 1, 1), device=device)

    # Combine masks
    type_is_attr = masking_type[:, None, None] < 1 / 3
    type_is_event = (masking_type[:, None, None] >= 1 / 3) & (
        masking_type[:, None, None] < 2 / 3
    )
    type_is_pos = masking_type[:, None, None] >= 2 / 3

    mask = (
        type_is_attr * attr_mask + type_is_event * event_mask + type_is_pos * pos_mask
    )

    mask = mask & ~known_mask

    # Create output tensor
    output = x.clone()

    # Apply masking
    full_mask_expanded = is_full_mask[:, None, None, None].expand(
        -1, num_events, num_attributes, vocab_size
    )
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, vocab_size)

    output = torch.where(
        full_mask_expanded & mask_expanded, torch.ones_like(output), output
    )

    sparse_positions = ~is_full_mask[:, None, None, None] & mask_expanded
    if sparse_positions.any():
        vocab_mask = torch.rand(
            batch_size, num_events, num_attributes, vocab_size, device=device
        ) < torch.rand((batch_size, 1, 1, 1), device=device)
        output = torch.where(
            sparse_positions & vocab_mask, torch.ones_like(output), output
        )

    return output, mask


def analyze_masking_statistics(
    num_trials=1000, batch_size=4, num_events=50, num_attributes=3, vocab_size=100
):
    """Analyze masking statistics over multiple trials."""
    stats = {"unknown_ratio": [], "fully_unknown_ratio": [], "partly_known_ratio": []}

    for _ in tqdm(range(num_trials), desc="Running trials"):
        # Create random one-hot batch
        x = torch.zeros(batch_size, num_events, num_attributes, vocab_size)
        indices = torch.randint(vocab_size, (batch_size, num_events, num_attributes))
        x.scatter_(-1, indices.unsqueeze(-1), 1)

        # Apply superposition
        masked_x, mask = apply_superposition(x)

        # Calculate statistics
        total_positions = num_events * num_attributes
        unknown_positions = mask.float().sum(dim=(1, 2))
        fully_unknown = (masked_x.sum(dim=-1) == vocab_size).float().sum(dim=(1, 2))
        partly_known = unknown_positions - fully_unknown

        stats["unknown_ratio"].append(
            (unknown_positions / total_positions).cpu().numpy()
        )
        stats["fully_unknown_ratio"].append(
            (fully_unknown / total_positions).cpu().numpy()
        )
        stats["partly_known_ratio"].append(
            (partly_known / total_positions).cpu().numpy()
        )

    # Convert to numpy arrays
    for key in stats:
        stats[key] = np.array(stats[key])

    return stats


def plot_masking_statistics(stats, save_dir="figures"):
    """Plot and save masking statistics distributions."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ["Unknown Positions", "Fully Unknown", "Partly Known"]
    keys = ["unknown_ratio", "fully_unknown_ratio", "partly_known_ratio"]

    for ax, title, key in zip(axes, titles, keys):
        data = stats[key].flatten()
        ax.hist(data, bins=50, density=True)
        ax.set_title(f"{title}\nMean: {data.mean():.3f}\nStd: {data.std():.3f}")
        ax.set_xlabel("Ratio")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "masking_statistics.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def visualize_mask_samples(
    num_samples=20, num_events=50, num_attributes=3, vocab_size=100, save_dir="figures"
):
    """Create and save visualizations of mask patterns."""
    os.makedirs(save_dir, exist_ok=True)

    # Create samples
    x = torch.zeros(num_samples, num_events, num_attributes, vocab_size)
    indices = torch.randint(vocab_size, (num_samples, num_events, num_attributes))
    x.scatter_(-1, indices.unsqueeze(-1), 1)

    # Apply superposition
    masked_x, mask = apply_superposition(x)

    # Create visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        # Create a combined visualization showing both mask and masking type
        viz_data = torch.zeros(num_events, num_attributes, 3)  # RGB channels

        # Red channel: original mask
        viz_data[:, :, 0] = mask[i]

        # Green channel: fully masked positions
        fully_masked = masked_x[i].sum(dim=-1) == vocab_size
        viz_data[:, :, 1] = fully_masked

        # Blue channel: partly masked positions
        partly_masked = mask[i] & ~fully_masked
        viz_data[:, :, 2] = partly_masked

        ax.imshow(viz_data.cpu().numpy())
        ax.set_title(f"Sample {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "mask_samples.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def print_statistics(stats):
    """Print detailed statistics."""
    for key in ["unknown_ratio", "fully_unknown_ratio", "partly_known_ratio"]:
        data = stats[key].flatten()
        print(f"\n{key.replace('_', ' ').title()}:")
        print(f"  Mean: {data.mean():.3f}")
        print(f"  Std:  {data.std():.3f}")
        print(f"  Min:  {data.min():.3f}")
        print(f"  Max:  {data.max():.3f}")
        print(f"  Median: {np.median(data):.3f}")


def test_superposition():
    """Test function to verify superposition masking behavior and save visualizations."""
    print(f"Analyzing masking statistics over 1000 trials...")
    print(f"Settings: num_events=50, num_attributes=3, vocab_size=100\n")

    # Run statistical analysis
    stats = analyze_masking_statistics()

    # Print statistics
    print_statistics(stats)

    # Save plots
    print("\nSaving distribution plots to 'figures/masking_statistics.png'...")
    plot_masking_statistics(stats)

    print("Saving sample visualizations to 'figures/mask_samples.png'...")
    visualize_mask_samples()

    print("\nAnalysis complete. Check the 'figures' directory for visualizations.")


if __name__ == "__main__":
    test_superposition()
