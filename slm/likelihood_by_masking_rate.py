import torch
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from train import TrainingWrapper
from data import MidiDataset
import pandas as pd
from paper_checkpoints import CHECKPOINTS
from tqdm import tqdm

# Configuration
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
SEED = 42

# Analysis parameters
N_EXAMPLES = 1  # Number of examples to analyze
SUPERPOSITION_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]  # Levels of superposition to test
N_MASKING_RATIOS = 21  # Number of masking ratio points (0% to 100%)
OUTPUT_DIR = Path("./analysis_results")


def setup_model(checkpoint_path):
    model = TrainingWrapper.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
    model.eval()
    return model


def load_test_dataset(tokenizer):
    mmd_4bar_filter_fn = lambda x: "n_bars=4" in x
    sm_filter_fn = lambda sm: not any(
        track.program == 0 and not track.is_drum and "piano" not in track.name.lower()
        for track in sm.tracks
    )

    test_ds = MidiDataset(
        cache_path="./data/mmd_loops/tst_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn,
        genre_list=tokenizer.config["tags"],
        tokenizer=tokenizer,
        min_notes=16,
        max_notes=tokenizer.config["max_notes"],
        use_random_shift=False,
        sm_filter_fn=sm_filter_fn,
    )
    return test_ds


def create_cumulative_superposition_masks(
    test_tokens, masking_positions, superposition_ratios, vocab_size
):
    """
    Creates a series of masks where each mask builds upon the previous one,
    adding more superposition at the same positions
    """
    # Convert tokens to one-hot vectors
    n_positions = len(test_tokens)
    base_mask = torch.zeros((n_positions, vocab_size))
    base_mask[torch.arange(n_positions), test_tokens] = 1.0

    masks = []
    current_mask = base_mask.clone()

    # For each superposition ratio
    for ratio in superposition_ratios:
        mask = current_mask.clone()

        # For each position to be masked
        for pos in masking_positions:
            # Create uniform distribution
            uniform_probs = torch.ones(vocab_size) / vocab_size

            # Interpolate between one-hot and uniform
            mask[pos] = (1 - ratio) * base_mask[pos] + ratio * uniform_probs

        masks.append(mask)
        current_mask = mask  # Update current mask for next iteration

    return masks


def analyze_model(model_name, checkpoint_path, test_examples):
    print(f"\nAnalyzing model: {model_name}")

    # Setup
    model = setup_model(checkpoint_path)
    masking_ratios = np.linspace(0, 1, N_MASKING_RATIOS)  # 0% to 100%
    vocab_size = len(model.tokenizer.vocab)

    results = []

    # Process each example
    for example_idx, test_tokens in enumerate(
        tqdm(test_examples, desc="Processing examples")
    ):
        with torch.no_grad():
            # Convert to tensor if needed
            if not isinstance(test_tokens, torch.Tensor):
                test_tokens = torch.tensor(test_tokens)

            # Get total number of positions
            total_positions = len(test_tokens)

            # Create fixed random order of positions for this example
            all_positions = list(range(total_positions))
            random.shuffle(all_positions)

            # For each masking ratio
            for mask_ratio in masking_ratios:
                # Calculate number of positions to mask for this ratio
                n_positions = int(mask_ratio * total_positions)
                masking_positions = all_positions[:n_positions]

                # Create cumulative superposition masks for all ratios
                superposition_masks = create_cumulative_superposition_masks(
                    test_tokens, masking_positions, SUPERPOSITION_RATIOS, vocab_size
                )

                # Calculate log likelihood for each superposition ratio
                for sup_ratio, modified_mask in zip(
                    SUPERPOSITION_RATIOS, superposition_masks
                ):
                    log_prob = model.model.conditional_log_likelihood(
                        test_tokens.to(DEVICE), modified_mask.to(DEVICE)
                    ).item()

                    results.append(
                        {
                            "model": model_name,
                            "example_idx": example_idx,
                            "superposition_ratio": sup_ratio,
                            "masking_ratio": mask_ratio,
                            "n_masked_positions": n_positions,
                            "log_likelihood": log_prob,
                        }
                    )

    return results


def plot_results(results_df, output_dir):
    # Plot for each model
    for model_name in results_df["model"].unique():
        model_data = results_df[results_df["model"] == model_name]

        # Create main plot
        plt.figure(figsize=(12, 8))

        # Calculate mean and std for each superposition/masking ratio combination
        for sup_ratio in sorted(model_data["superposition_ratio"].unique()):
            ratio_data = model_data[model_data["superposition_ratio"] == sup_ratio]

            # Group by masking ratio and calculate statistics
            grouped = ratio_data.groupby("masking_ratio")["log_likelihood"]
            means = grouped.mean()
            stds = grouped.std()

            # Plot mean line with error bands
            plt.plot(
                means.index * 100,
                means.values,
                marker="o",
                label=f"Superposition {sup_ratio:.2f}",
            )
            plt.fill_between(
                means.index * 100,
                means.values - stds.values,
                means.values + stds.values,
                alpha=0.2,
            )

        plt.xlabel("Masking Ratio (%)")
        plt.ylabel("Log Likelihood")
        plt.title(
            f"Log Likelihood vs Masking Ratio - {model_name}\n(Mean ± Std over {N_EXAMPLES} examples)"
        )
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.7)

        # Save plot
        plt.savefig(
            output_dir / f"{model_name}_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def main():
    # Set random seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Setup output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load first model to get tokenizer and dataset
    first_model = setup_model(next(iter(CHECKPOINTS.values())))
    test_dataset = load_test_dataset(first_model.tokenizer)

    # Get random examples
    example_indices = random.sample(range(len(test_dataset)), N_EXAMPLES)
    test_examples = [test_dataset[idx]["token_ids"] for idx in example_indices]

    # Analyze each model
    all_results = []
    for model_name, checkpoint_path in CHECKPOINTS.items():
        results = analyze_model(model_name, checkpoint_path, test_examples)
        all_results.extend(results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    results_df.to_csv(OUTPUT_DIR / "analysis_results.csv")

    # Plot results
    plot_results(results_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
