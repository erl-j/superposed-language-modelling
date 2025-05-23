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
N_EXAMPLES = 50  # Number of examples to analyze
N_CONFOUNDERS = [1, 2, 4, 8, 16, 32, 64, 128, 256, "full"]  # Number of confounders to test
N_MASKING_RATIOS = 21  # Number of masking ratio points (0% to 100%)
OUTPUT_DIR = Path("./analysis_results_confounders")


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

    # set seeds
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

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

        for n_confounders in tqdm(N_CONFOUNDERS, desc="Number of confounders"):
            print(f"Number of confounders: {n_confounders}")
            # set random seed
            torch.manual_seed(SEED)
            
            # Create superposition mask with fixed number of confounders
            candidate_confounders = model.syntax_mask.to(x.device) - x.to(x.device)

            if n_confounders == "full":
                # Use all possible confounders
                confounders = candidate_confounders
            else:
                # sample n confounders per position
                confounders = torch.rand_like(x) * candidate_confounders
                # get top k values and indices
                values, indices = confounders.topk(n_confounders, dim=-1)
                # create zero tensor of same shape as x
                confounders = torch.zeros_like(x)
                # scatter 1s at the top k indices
                confounders.scatter_(-1, indices, 1)

            print(confounders.shape)

            superposition_mask = x + confounders

            assert (superposition_mask.sum(dim=-1) >= 1).all()

            for masking_ratio in tqdm(masking_ratios, desc="Masking ratios", leave=False):
                print(f"Masking ratio: {masking_ratio}")
                # Update position mask
                position_mask = torch.rand(batch_size, n_events, n_attributes, 1, device=DEVICE) < masking_ratio

                # Combine masks
                prior = torch.clamp(x + position_mask * superposition_mask, 0, 1)
                prior = prior / prior.sum(dim=-1, keepdim=True)

                # Calculate log likelihood
                log_probs = model.model.conditional_log_likelihood(x.clone(), prior, enforce_constraint=enforce_constraint)

                # Store results
                results.append({
                    "model": model_name,
                    "n_confounders": n_confounders,
                    "masking_ratio": masking_ratio,
                    "log_likelihood": log_probs.mean().item(),
                })

    return results

def plot_results(results_df, output_dir):
    # Create a plot for each number of confounders
    for n_conf in results_df["n_confounders"].unique():
        plt.figure(figsize=(12, 8))

        # Plot each model's data for this number of confounders
        for model_name in results_df["model"].unique():
            # Filter data for current model and number of confounders
            model_data = results_df[
                (results_df["model"] == model_name)
                & (results_df["n_confounders"] == n_conf)
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
                alpha=0.2,
            )

        plt.xlabel("Masking Ratio (%)")
        plt.ylabel("Log Likelihood")
        plt.title(
            f"Log Likelihood vs Masking Ratio\nNumber of Confounders: {n_conf}\n"
            f"(Mean ± Std over {N_EXAMPLES} examples)"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, which="both", linestyle="--", alpha=0.7)

        # Save plot
        plt.savefig(
            output_dir / f"confounders_{n_conf}_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

def plot_combined_results(results_df, output_dir):
    plt.figure(figsize=(12, 8))
    
    # Define color scheme
    mlm_color = '#1f77b4'  # Blue for MLM
    slm_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for SLM variants
    
    # Sort confounders properly
    def confounder_sort_key(x):
        if x == "full":
            return float('inf')
        try:
            return float(x)
        except ValueError:
            return float('inf')
    
    n_confounders = sorted(results_df["n_confounders"].unique(), key=confounder_sort_key)
    
    # Plot each model's data
    for model_name in results_df["model"].unique():
        for i, n_conf in enumerate(n_confounders):
            # Filter data for current model and number of confounders
            model_data = results_df[
                (results_df["model"] == model_name)
                & (results_df["n_confounders"] == n_conf)
            ]

            # Group by masking ratio and calculate statistics
            grouped = model_data.groupby("masking_ratio")["log_likelihood"]
            means = grouped.mean()
            stds = grouped.std()

            # Choose color based on model type
            if model_name.startswith("mlm"):
                color = mlm_color
                linestyle = '-' if "w_constraint" in model_name else '--'
            else:
                color = slm_colors[i % len(slm_colors)]
                linestyle = '-'

            # Plot mean line with error bands
            plt.plot(
                means.index * 100,
                means.values,
                marker="o",
                label=f"{model_name} (conf={n_conf})",
                color=color,
                linestyle=linestyle,
                alpha=0.8,
            )
            plt.fill_between(
                means.index * 100,
                means.values - stds.values,
                means.values + stds.values,
                color=color,
                alpha=0.2,
            )

    plt.xlabel("Masking Ratio (%)")
    plt.ylabel("Log Likelihood")
    plt.title(f"Log Likelihood vs Masking Ratio\n(Mean ± Std over {N_EXAMPLES} examples)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)

    # Save plot
    plt.savefig(
        output_dir / "combined_confounders_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def plot_subplot_results(results_df, output_dir):
    # Define color scheme
    mlm_color = '#1f77b4'  # Blue for MLM
    slm_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for SLM variants
    
    # Sort confounders properly
    def confounder_sort_key(x):
        if x == "full":
            return float('inf')
        try:
            return float(x)
        except ValueError:
            return float('inf')
    
    n_confounders = sorted(results_df["n_confounders"].unique(), key=confounder_sort_key)
    n_cols = (len(n_confounders) + 1) // 2  # Ceiling division
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
    axes = axes.flatten()
    
    # Get global y-axis limits
    y_min = results_df["log_likelihood"].min()
    y_max = results_df["log_likelihood"].max()
    
    # Create handles and labels for legend
    legend_handles = []
    legend_labels = []
    
    # Plot each number of confounders in a subplot
    for idx, n_conf in enumerate(n_confounders):
        ax = axes[idx]
        
        # Plot MLM systems
        mlm_data = results_df[
            (results_df["model"].str.startswith("mlm"))
            & (results_df["n_confounders"] == n_conf)
        ]
        for i, model_name in enumerate(mlm_data["model"].unique()):
            model_subset = mlm_data[mlm_data["model"] == model_name]
            grouped = model_subset.groupby("masking_ratio")["log_likelihood"]
            means = grouped.mean()
            stds = grouped.std()
            
            label = "MLM (w/ constraint)" if "w_constraint" in model_name else "MLM (w/o constraint)"
            line = ax.plot(means.index * 100, means.values, 
                   color=mlm_color, 
                   marker='o',
                   linestyle='-' if "w_constraint" in model_name else '--',
                   alpha=0.8)
            ax.fill_between(means.index * 100,
                          means.values - stds.values,
                          means.values + stds.values,
                          color=mlm_color,
                          alpha=0.2)
            
            if idx == 0:  # Only add to legend once
                legend_handles.extend(line)
                legend_labels.append(label)
        
        # Plot SLM systems
        slm_data = results_df[
            (results_df["model"].str.startswith("slm"))
            & (results_df["n_confounders"] == n_conf)
        ]
        for i, model_name in enumerate(slm_data["model"].unique()):
            model_subset = slm_data[slm_data["model"] == model_name]
            grouped = model_subset.groupby("masking_ratio")["log_likelihood"]
            means = grouped.mean()
            stds = grouped.std()
            
            slm_type = model_name.replace("slm_", "").replace("_150epochs", "")
            line = ax.plot(means.index * 100, means.values,
                   color=slm_colors[i],
                   marker='o',
                   label=f"SLM ({slm_type})",
                   alpha=0.8)
            ax.fill_between(means.index * 100,
                          means.values - stds.values,
                          means.values + stds.values,
                          color=slm_colors[i],
                          alpha=0.2)
            
            if idx == 0:  # Only add to legend once
                legend_handles.extend(line)
                legend_labels.append(f"SLM ({slm_type})")
        
        ax.set_xlabel("Masking Ratio (%)")
        ax.set_ylabel("Log Likelihood")
        ax.set_title(f"Number of Confounders: {n_conf}")
        ax.grid(True, which="both", linestyle="--", alpha=0.7)
        ax.set_ylim(y_min, y_max)  # Set consistent y-axis limits
    
    # Remove empty subplots if any
    for idx in range(len(n_confounders), len(axes)):
        fig.delaxes(axes[idx])
    
    # Add single legend to the figure
    fig.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    plt.tight_layout()
    plt.savefig(output_dir / "confounders_subplot_analysis.png", dpi=300, bbox_inches="tight")
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
    plot_subplot_results(results_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
