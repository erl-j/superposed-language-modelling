# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import symusic
from util import piano_roll
import muspy
import tqdm
import pandas as pd

# TODO: remove layers from structural analysis

# load csv

# Load ground truth examples
base_path = Path("../artefacts/applications")

csv_path = base_path / "records.csv"

# Load records from CSV
records = pd.read_csv(csv_path)

# plot mean log likelhood for each model and task
models = records.model.unique()
tasks = records.task.unique()

for task in tasks:
    for model in models:
        mean_ll = records[(records.model == model) & (records.task == task)].log_probs.mean()
        print(f"Task: {task}, Model: {model}, Mean Log Likelihood: {mean_ll}")

#%%

ground_truth_path = base_path / "ground_truth"
ground_truth_files = list(ground_truth_path.glob("*.mid"))

# Dynamically discover models and tasks
models = [
    d.name for d in base_path.iterdir() if d.is_dir() and d.name != "ground_truth"
]
# Get tasks from first model's directory (assuming all models have same tasks)
first_model_path = base_path / models[0]
tasks = [d.name for d in first_model_path.iterdir() if d.is_dir()]

print(f"Discovered models: {models}")
print(f"Discovered tasks: {tasks}")

# Configuration
n_examples = len(ground_truth_files)

# Create records for ground truth using list comprehension
ground_truth_midis = [symusic.Score(f) for f in ground_truth_files]
records = [
    {
        "midi": midi,
        "piano_roll": piano_roll(midi, 96),
        "path": path,
        "task": "ground_truth",
        "model": "ground_truth",
        "sample_index" : int(path.split(".mid")[0].split("_")[-1])
    }
    for midi, path in zip(ground_truth_midis, ground_truth_files)
]

# Add records for each generated midi using list comprehension
records.extend(
    [
        {
            "midi": symusic.Score(file),
            "piano_roll": piano_roll(symusic.Score(file), 96),
            "path": file,
            "task": task,
            "model": model,
        }
        for task in tasks
        for model in models
        for file in (base_path / model / task).glob("*.mid")
    ]
)

# Calculate pitch in scale rate using list comprehension
records = [
    {
        **record,
        "metric/scale_consitency": muspy.metrics.scale_consistency(
            muspy.read_midi(record["path"])
        ),
        "metric/n_instruments" : len(record["midi"].tracks),
        "metric/polyphony_rate" : muspy.metrics.polyphony_rate(
            muspy.read_midi(record["path"])
        ),
        "metric/pitch_class_entropy" : muspy.metrics.pitch_class_entropy(
            muspy.read_midi(record["path"])
        ),
        "metric/polyphony": muspy.metrics.polyphony(
            muspy.read_midi(record["path"])
        ),
        "metric/n_pitches_used": muspy.metrics.n_pitches_used(
            muspy.read_midi(record["path"])
        ),
        "metric/n_pitch_classes_used": muspy.metrics.n_pitch_classes_used(
            muspy.read_midi(record["path"])
        ),
        "metric/pitch_entropy": muspy.metrics.pitch_entropy(
            muspy.read_midi(record["path"])
        )}
    for record in records
]

print(f"Processed {len(records)} records")

#%%

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Convert records to DataFrame for easier plotting
df = pd.DataFrame(records)

# Get all metrics columns
metric_cols = [col for col in df.columns if col.startswith("metric/")]

# Create violin plots for each task
for task in tasks:
    # Filter data for current task
    task_df = df[df["task"] == task]

    # Calculate number of metrics and set up subplot grid
    n_metrics = len(metric_cols)
    n_rows = (n_metrics + 2) // 3  # 3 plots per row, round up
    n_cols = min(3, n_metrics)

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f"Metric Distributions for Task: {task}", size=16, y=1.02)

    # Flatten axes array for easier iteration
    if n_rows > 1:
        axes = axes.flatten()
    elif n_rows == 1 and n_cols == 1:
        axes = [axes]

    # Create violin plot for each metric
    for idx, metric in enumerate(metric_cols):
        if idx < len(axes):  # Ensure we don't exceed number of subplots
            sns.violinplot(data=task_df, x="model", y=metric, ax=axes[idx])
            # Rotate x-axis labels for better readability
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
            # Clean up metric name for title
            metric_name = metric.replace("metric/", "").replace("_", " ").title()
            axes[idx].set_title(metric_name)
            axes[idx].set_xlabel("")

    # Remove any empty subplots
    for idx in range(len(metric_cols), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    plt.show()
# Calculate summary statistics for each metric by task and model
summary_stats = []
for task in tasks:
    task_df = df[df["task"] == task]
    for model in models + ["ground_truth"]:
        model_df = task_df[task_df["model"] == model]
        stats = {"task": task, "model": model}
        for metric in metric_cols:
            stats[f"{metric}_mean"] = model_df[metric].mean()
            stats[f"{metric}_std"] = model_df[metric].std()
        summary_stats.append(stats)

# Convert summary statistics to DataFrame
summary_df = pd.DataFrame(summary_stats)

# Print summary table
print("\nSummary Statistics:")
print(summary_df.to_string())

# Optionally save summary statistics to CSV
# summary_df.to_csv("metric_summary.csv", index=False)
    
#%%

df = pd.DataFrame(records)

# For better display formatting
pd.set_option("display.precision", 3)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.colheader_justify", "left")
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)

# Get all metric columns
metric_cols = [col for col in df.columns if col.startswith("metric/")]

# Create metrics dictionary with clean names
metrics = {
    col.replace("metric/", "").replace("_", " ").title(): df.groupby(["model", "task"])[
        col
    ].mean()
    for col in metric_cols
}

# Combine into a single table
results = pd.concat(metrics, axis=1)

# Display
try:
    from IPython.display import display

    print("\nMusic Generation Model Metrics:")
    display(
        results.style.format(precision=3).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("text-align", "left"), ("padding", "8px")],
                },
                {
                    "selector": "td",
                    "props": [("text-align", "left"), ("padding", "8px")],
                },
            ]
        )
    )
except ImportError:
    print("\nMusic Generation Model Metrics:")
    print("=" * 80)
    print(results)

#%%
# take mean piano roll for each model and task and show
# crop piano rolls to min length
for model in models:
    for task in tasks:
        piano_rolls = df[(df.model == model) & (df.task == task)].piano_roll
        min_length = min([pr.shape[1] for pr in piano_rolls])
        mean_piano_roll = np.mean([pr[:, :min_length] for pr in piano_rolls], axis=0)
        plt.imshow(mean_piano_roll, aspect="auto", origin="lower", cmap="Blues")
        plt.title(f"{model} {task}")
        plt.show()
#%%

# n_cols = 1 + len(models)  # ground truth + each model
# n_rows = n_examples
# figsize = (20, 4 * n_examples)  # Adjust height based on number of examples
# tpq = 96

# # Create visualization for each task
# for task in tasks:
#     print(f"\nViewing task: {task}")

#     # Create figure
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
#     if n_rows == 1:
#         axes = axes.reshape(1, -1)

#     # Plot ground truth in first column
#     for row, sm in enumerate(ground_truth_midis):
#         ax = axes[row, 0]
#         roll = piano_roll(sm, tpq)
#         im = ax.imshow(roll, aspect="auto", origin="lower", cmap="Blues")
#         if row == 0:  # Add column title for ground truth
#             ax.set_title("Ground Truth", pad=10)
#         ax.set_xlabel("Time (ticks)")
#         ax.set_ylabel("MIDI Note")

#     # Plot each model's output in subsequent columns
#     for col, model in enumerate(models, start=1):
#         model_path = base_path / model / task
#         model_files = list(model_path.glob("*.mid"))
#         model_midis = [symusic.Score(f) for f in model_files]

#         for row, sm in enumerate(model_midis):
#             ax = axes[row, col]
#             roll = piano_roll(sm, tpq)
#             im = ax.imshow(roll, aspect="auto", origin="lower", cmap="Blues")
#             if row == 0:  # Add column title for each model
#                 ax.set_title(model, pad=10)
#             ax.set_xlabel("Time (ticks)")
#             if col > 0:  # Remove redundant y-labels
#                 ax.set_ylabel("")
#                 ax.set_yticks([])

#     plt.suptitle(f"Task: {task}", size=16, y=1.02)
#     plt.tight_layout()
#     plt.show()

# %%
