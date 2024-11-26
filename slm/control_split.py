import torch
import symusic
import numpy as np
import einops
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore


def get_piano_roll(sm):
    """Convert symusic MIDI to piano roll representation."""
    sm = sm.copy()
    # resample to 4 ticks per quarter note, use int8
    pr = (
        sm.resample(2)
        .pianoroll(modes=["frame"], pitch_range=[20, 100])[0]
        .sum(axis=0)
        .astype("float32")
    )
    return pr


def load_and_process_data(data_path):
    """Load and preprocess MIDI records."""
    records = torch.load(data_path)
    # flatten
    records_flat = [x for record in records for x in record]
    # filter by midi with n_bars=4.0 in path
    records_flat = [x for x in records_flat if "n_bars=4.0" in x["path"]]

    # Add piano roll representation
    records_flat = [
        {**x, "pr": get_piano_roll(x["midi"])}
        for x in tqdm(records_flat, desc="Processing MIDI files")
    ]
    return records_flat


def compute_cross_distance_matrix(
    trn_records, val_records, device="cuda", batch_size=1000
):
    """
    Compute distance matrix between training and validation sets using GPU batching.

    Args:
        trn_records: List of training records
        val_records: List of validation records
        device: 'cuda' or 'cpu'
        batch_size: Size of batches for GPU computation
    """
    # Extract piano rolls
    trn_piano_rolls = [x["pr"] for x in trn_records]
    val_piano_rolls = [x["pr"] for x in val_records]

    # Get min length and crop
    min_len = min(
        min(x.shape[1] for x in trn_piano_rolls),
        min(x.shape[1] for x in val_piano_rolls),
    )

    # Crop all sequences to min_len
    trn_piano_rolls = [x[:, :min_len] for x in trn_piano_rolls]
    val_piano_rolls = [x[:, :min_len] for x in val_piano_rolls]

    # Convert to tensor and reshape
    trn_tensor = torch.tensor(trn_piano_rolls)
    val_tensor = torch.tensor(val_piano_rolls)

    trn_tensor = einops.rearrange(trn_tensor, "n p t -> n (p t)")
    val_tensor = einops.rearrange(val_tensor, "n p t -> n (p t)")

    # Move to specified device
    trn_tensor = trn_tensor.to(device)
    val_tensor = val_tensor.to(device)

    n_val, feat_dim = val_tensor.shape
    n_trn = trn_tensor.shape[0]

    # Initialize distance matrix on CPU to save GPU memory
    distance_matrix = torch.zeros((n_val, n_trn), device="cpu")

    # Compute distance matrix in batches
    with tqdm(total=n_val, desc="Computing distances") as pbar:
        for i in range(0, n_val, batch_size):
            # Get current batch
            batch_end = min(i + batch_size, n_val)
            val_batch = val_tensor[i:batch_end]

            # Compute distances for current batch
            batch_distances = torch.cdist(val_batch, trn_tensor, p=1)

            # Move batch results to CPU
            distance_matrix[i:batch_end] = batch_distances.cpu()

            pbar.update(batch_end - i)

    return distance_matrix


def find_potential_leakage(
    distance_matrix, trn_records, val_records, threshold_percentile=1
):
    """Find pairs of samples between train and val that are suspiciously similar."""
    # Get the threshold value based on percentile
    flat_sim = distance_matrix.flatten()
    threshold = np.percentile(flat_sim.numpy(), threshold_percentile)

    # Find pairs below threshold
    suspicious_pairs = []
    n_val, n_trn = distance_matrix.shape
    for i in range(n_val):
        for j in range(n_trn):
            if distance_matrix[i, j] < threshold:
                suspicious_pairs.append(
                    {
                        "val_idx": i,
                        "trn_idx": j,
                        "distance": distance_matrix[i, j].item(),
                        "val_path": val_records[i]["path"],
                        "trn_path": trn_records[j]["path"],
                    }
                )

    return sorted(suspicious_pairs, key=lambda x: x["distance"])


def plot_distance_distribution(distance_matrix):
    """Plot the distribution of distance scores."""
    plt.figure(figsize=(12, 6))

    # Create main histogram
    plt.subplot(1, 2, 1)
    plt.hist(distance_matrix.flatten().numpy(), bins=50, density=True)
    plt.title("Distribution of Cross-distance Scores")
    plt.xlabel("L1 Distance")
    plt.ylabel("Density")
    plt.grid(True)

    # Create log-scale histogram to better see the tail
    plt.subplot(1, 2, 2)
    plt.hist(distance_matrix.flatten().numpy(), bins=50, density=True)
    plt.yscale("log")
    plt.title("Distribution (Log Scale)")
    plt.xlabel("L1 Distance")
    plt.ylabel("Density (log)")
    plt.grid(True)

    plt.tight_layout()
    return plt


def main(device="cuda", batch_size=1000):
    # Check if CUDA is available when device is set to 'cuda'
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    print(f"Using device: {device}")

    # Load training and validation data
    base_path = "./data/mmd_loops"
    val_path = f"{base_path}/val_midi_records_unique_pr.pt"
    trn_path = f"{base_path}/trn_midi_records_unique_pr.pt"

    print("Loading and processing validation data...")
    val_records = load_and_process_data(val_path)
    print("Loading and processing training data...")
    trn_records = load_and_process_data(trn_path)

    print(f"\nDataset sizes:")
    print(f"Training samples: {len(trn_records)}")
    print(f"Validation samples: {len(val_records)}")

    # Compute cross-distance matrix
    print("\nComputing cross-distance matrix...")
    distance_matrix = compute_cross_distance_matrix(
        trn_records, val_records, device=device, batch_size=batch_size
    )

    # Basic statistics
    print(f"\ndistance matrix shape: {distance_matrix.shape}")
    print(f"Min distance: {distance_matrix.min().item():.2f}")
    print(f"Max distance: {distance_matrix.max().item():.2f}")
    print(f"Mean distance: {distance_matrix.mean().item():.2f}")

    # Find potential leakage
    print("\nFinding suspicious pairs...")
    # suspicious_pairs = find_potential_leakage(
    #     distance_matrix, trn_records, val_records, threshold_percentile=1
    # )

    # Print results
    # print(f"\nFound {len(suspicious_pairs)} suspicious pairs (top 10 shown):")
    # for pair in suspicious_pairs[:10]:
    #     print(f"\ndistance score: {pair['distance']:.2f}")
    #     print(f"Validation file: {pair['val_path']}")
    #     print(f"Training file: {pair['trn_path']}")

    # Plot distance distribution
    print("\nPlotting distance distribution...")
    plt = plot_distance_distribution(distance_matrix)
    plt.savefig("distance_distribution.png")
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:6",
        choices=["cuda:6", "cpu"],
        help="Device to use for computation",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for GPU computation"
    )
    args = parser.parse_args()

    main(device=args.device, batch_size=args.batch_size)
