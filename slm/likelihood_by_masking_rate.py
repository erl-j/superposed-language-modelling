import torch
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from train import TrainingWrapper
from data import MidiDataset
import pandas as pd
from PAPER_CHECKPOINTS import CHECKPOINTS
from tqdm import tqdm
from masking import simple_superposition,random_superposition

# Configuration
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
SEED = 0

ENFORCE_CONSTRAINT = True
# Analysis parameters
N_EXAMPLES = 3  # Number of examples to analyze
SUPERPOSITION_RATIOS = [0.25, 0.5, 0.75, 1.0]  # Levels of superposition to test
N_MASKING_RATIOS = 5  # Number of masking ratio points (0% to 100%)
OUTPUT_DIR = Path("./analysis_results_5")


def setup_model(checkpoint_path):
    model = TrainingWrapper.load_from_checkpoint(checkpoint_path, map_location=DEVICE, weights_only=False)
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
        n_bars=4,
    )
    return test_ds

def analyze_model(model_name, enforce_constraint, checkpoint_path, x):



    print(f"\nAnalyzing model: {model_name}")

    # Setup
    model = setup_model(checkpoint_path)
    masking_ratios = np.linspace(0, 1.0, N_MASKING_RATIOS)  # 0% to 100%

    x = torch.nn.functional.one_hot(x, model.model.vocab_size).float()
    # Move to device
    x = x.to(DEVICE)

    print(x.shape)

    results = []

    with torch.no_grad():
        # Get dimensions
        batch_size, n_events, n_attributes, vocab_size = x.shape

        for superposition_ratio in tqdm(
            SUPERPOSITION_RATIOS, desc="Superposition ratios"
        ):  
            print(superposition_ratio)
            # set random seed
            # set seeds
            torch.manual_seed(SEED)
            random.seed(SEED)
            np.random.seed(SEED)
            # Create superposition mask
            
            superposition_mask = random_superposition(x, syntax_mask = model.syntax_mask, mode="shared_rate", ratio=(1-superposition_ratio))

            is_unknown = (superposition_mask > 0).sum(dim=-1) > 1
            print(f"Average of is unknown: {is_unknown.float().mean().item()}")

            for masking_ratio in tqdm(
                masking_ratios, desc="Masking ratios", leave=False
            ):
                print(f"Masking ratio: {masking_ratio}")
                # Update position mask
                position_mask = torch.rand(
                    batch_size, n_events, n_attributes, 1, device=DEVICE
                ) < masking_ratio

                # Combine masks

                prior = torch.clamp(x + position_mask * superposition_mask, 0, 1)

                # average ones per column
                print(prior.sum(dim=-1, keepdim=True).mean())

                prior = prior / prior.sum(dim=-1, keepdim=True)

                # Calculate log likelihood
                log_probs = model.model.conditional_log_likelihood(x.clone(), prior, enforce_constraint=enforce_constraint)

                # Store results for each example in batch
                results.append(
                    {
                        "model": model_name,
                        "superposition_ratio": superposition_ratio,
                        "masking_ratio": masking_ratio,
                        "log_likelihood": log_probs.mean().item(),
                    }
                )

    return results

def plot_results(results_df, output_dir):
    # Create a plot for each superposition ratio
    for sup_ratio in sorted(results_df["superposition_ratio"].unique()):
        plt.figure(figsize=(12, 8))

        # Plot each model's data for this superposition ratio
        for model_name in results_df["model"].unique():
            # Filter data for current model and superposition ratio
            model_data = results_df[
                (results_df["model"] == model_name)
                & (results_df["superposition_ratio"] == sup_ratio)
            ]

            # Group by masking ratio and calculate statistics
            grouped = model_data.groupby("masking_ratio")["log_likelihood"]
            means = grouped.mean()
            stds = grouped.std()

            # Plot mean line with error bands
            plt.plot(
                means.index * 100,
                means.values,
                marker="o",
                label=f"{model_name}",
                alpha=0.8,
            )
            plt.fill_between(
                means.index * 100,
                means.values - stds.values,
                means.values + stds.values,
                alpha=0.8,
            )

        plt.xlabel("Masking Ratio (%)")
        plt.ylabel("Log Likelihood")
        plt.title(
            f"Log Likelihood vs Masking Ratio\nSuperposition Ratio: {sup_ratio:.2f}\n"
            f"(Mean ± Std over {N_EXAMPLES} examples)"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, which="both", linestyle="--", alpha=0.7)

        # Save plot
        plt.savefig(
            output_dir / f"superposition_{sup_ratio:.3f}_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

def plot_combined_results(results_df, output_dir):
    plt.figure(figsize=(12, 8))
    
    # Define line styles for superposition ratios
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    # Plot each model's data
    for model_name in results_df["model"].unique():
        for i, sup_ratio in enumerate(sorted(results_df["superposition_ratio"].unique())):
            # Filter data for current model and superposition ratio
            model_data = results_df[
                (results_df["model"] == model_name)
                & (results_df["superposition_ratio"] == sup_ratio)
            ]

            # Group by masking ratio and calculate statistics
            grouped = model_data.groupby("masking_ratio")["log_likelihood"]
            means = grouped.mean()
            stds = grouped.std()

            # Plot mean line with error bands
            plt.plot(
                means.index * 100,
                means.values,
                marker="o",
                label=f"{model_name} (sup={sup_ratio:.2f})",
                linestyle=line_styles[i],
                alpha=0.8,
            )
            plt.fill_between(
                means.index * 100,
                means.values - stds.values,
                means.values + stds.values,
                alpha=0.2,
            )

    plt.xlabel("Masking Ratio (%)")
    plt.ylabel("Log Likelihood")
    plt.title(f"Log Likelihood vs Masking Ratio\n(Mean ± Std over {N_EXAMPLES} examples)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)

    # Save plot
    plt.savefig(
        output_dir / "combined_superposition_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def visualize_masks(x, model, superposition_ratios, output_dir):
    """Visualize and save masks for different superposition ratios."""
    x = torch.nn.functional.one_hot(x, model.model.vocab_size).float()
    x = x.to(DEVICE)
    
    # Get dimensions
    batch_size, n_events, n_attributes, vocab_size = x.shape
    
    # Create a figure for each superposition ratio
    for sup_ratio in superposition_ratios:
        plt.figure(figsize=(15, 5))
        
        # Create superposition mask
        superposition_mask = random_superposition(x, syntax_mask=model.syntax_mask, 
                                               mode="variable_rate", ratio=sup_ratio)
        
        # Plot the mask for the first example in batch
        mask_vis = superposition_mask[0].sum(dim=-1).cpu().numpy()
        plt.imshow(mask_vis, cmap='viridis', aspect='auto')
        plt.colorbar(label='Sum of mask values')
        plt.title(f'Superposition Mask (ratio={sup_ratio:.2f})')
        plt.xlabel('Event Position')
        plt.ylabel('Attribute')
        
        # Save the visualization
        plt.savefig(output_dir / f'superposition_mask_{sup_ratio:.2f}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():

    systems = {
        "mlm_150epochs_w_constraint": {
            "checkpoint": CHECKPOINTS["mlm_150epochs"],
            "enforce_constraint": True
        },
        "mlm_150epochs_wo_constraint": {
            "checkpoint": CHECKPOINTS["mlm_150epochs"],
            "enforce_constraint": False
        },
        "slm_sparse_150epochs": {
            "checkpoint": CHECKPOINTS["slm_sparse_150epochs"], 
            "enforce_constraint": True
        },
        "slm_mixed_150epochs": {
            "checkpoint": CHECKPOINTS["slm_mixed_150epochs"],
            "enforce_constraint": True
        },
        "slm_full_150epochs": {
            "checkpoint": CHECKPOINTS["slm_full_150epochs"],
            "enforce_constraint": True
        }
    }

    # Set random seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Setup output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load first model to get tokenizer and dataset
    first_model = setup_model(next(iter(systems.values()))["checkpoint"])
    test_dataset = load_test_dataset(first_model.tokenizer)

    # Get random examples
    example_indices = random.sample(range(len(test_dataset)), N_EXAMPLES)
    test_examples = [test_dataset[idx]["token_ids"] for idx in example_indices]

    # make tensor
    x = torch.stack(test_examples).to(DEVICE)

    # Visualize masks before analysis
    visualize_masks(x, first_model, SUPERPOSITION_RATIOS, OUTPUT_DIR)
    
    # Analyze each model
    all_results = []
    for system_name, system_config in systems.items():
        checkpoint_path = system_config["checkpoint"]
        results = analyze_model(system_name, system_config["enforce_constraint"], checkpoint_path, x)
        all_results.extend(results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    results_df.to_csv(OUTPUT_DIR / "analysis_results.csv")

    # Plot results
    plot_results(results_df, OUTPUT_DIR)
    plot_combined_results(results_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
