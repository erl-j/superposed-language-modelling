#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import symusic
from util import piano_roll
import muspy
import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# TODO: remove layers from structural analysis

# load csv

# Load ground truth examples
base_path = Path("../artefacts/applications_250")

import glob

csv_paths = glob.glob(str(base_path / "**/*records.csv"), recursive=True)

print(f"Found {len(csv_paths)} records.csv files")

# Join all records in one step
records = pd.concat([
    pd.read_csv(path) 
    for path in csv_paths
], ignore_index=True)

# sort by model
records = records.sort_values(by=["model", "task"])



# models_to_keep = ["slm_sparse_150epochs", 
#                   "slm_full_150epochs", 
#                   "slm_mixed_150epochs",
#                 #   "slm_mixed_150epochs_event_order", 
#                   "mlm_150epochs", 
#                 #   "mlm_150epochs_event_order", 
#                   "ground_truth"]
models_to_keep = records.model.unique()
tasks_to_keep =records.task.unique() #["infill_middle", "replace_pitches_given_pitch_set", "unconditional", "ground_truth", "constrained_generation"]

records = records[records.model.isin(models_to_keep)]
records = records[records.task.isin(tasks_to_keep)]

# plot mean log likelhood for each model and task
models = records.model.unique()
tasks = records.task.unique()

for task in tasks:
    for model in models:
        mean_ll = records[(records.model == model) & (records.task == task)].log_probs.mean()
        print(f"Task: {task}, Model: {model}, Mean Log Likelihood: {mean_ll}")

for task in tasks:
    fig, axes = plt.subplots(len(models), 1, sharex=True, sharey=True)
    fig.suptitle(f"Log Likelihood Distribution for Task: {task}", size=16, y=1.02)
    
    # First calculate the global min and max for this task to set consistent bins
    task_data = records[records.task == task]
    min_ll = task_data.log_probs.min()
    max_ll = task_data.log_probs.max()
    bins = np.linspace(min_ll, max_ll, 21)  # 20 bins, 21 edges
    
    # Plot histograms using the same bins for all models
    for i, model in enumerate(models):
        model_ll = task_data[task_data.model == model].log_probs
        axes[i].hist(model_ll, bins=bins, alpha=0.5, label=model)
        axes[i].legend()
#%%

# inspect applications

# load fmd results
fmd_results = pd.read_json(base_path / "fmd_results.json")

# load into pandas dataframe
df = pd.DataFrame(fmd_results)

# keep only models

# df = df[df.model.isin(models_to_keep)]
df = df[df.task.isin(tasks_to_keep)]
# rename tasks
df["task"] = df["task"].str.replace("infill_middle", "replace_notes_within_box")
# crop model strings to before "_"
# df["model"] = df["model"].str.split("_").str[0]
# pinrt models
print(df.model.unique())

# sort by model and task
df = df.sort_values(by=["model", "task"])

pivot_df = df.pivot(index='model', columns='task', values='score')

# Print the table using pandas styling
print(pivot_df.to_string())

# Alternatively, for a prettier format:
display(pivot_df.style.format("{:.3f}"))

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
        "piano_roll": piano_roll(midi, 24, include_drums=False),
        "path": path,
        "task": "ground_truth",
        "model": "ground_truth",
        "sample_index" : int(str(path).split(".mid")[0].split("_")[-1])
    }
    for midi, path in zip(ground_truth_midis, ground_truth_files)
]

# Add records for each generated midi using list comprehension
records.extend(
    [
        {
            "midi": symusic.Score(file),
            "piano_roll": piano_roll(symusic.Score(file), 24, include_drums=False),
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



#%%


# Convert records to DataFrame for easier plotting
df = pd.DataFrame(records)


# sort by task and model 
df = df.sort_values(by=["model", "task", "sample_index"])

# remove "mlm_100epochs"
models_to_keep = [model for model in models if "mlm_100epochs" not in model]
# filter models and tasks
df = df[df.model.isin(models_to_keep)]
df = df[df.task.isin(tasks_to_keep)]

# rename tasks
df["task"] = df["task"].str.replace("infill_middle", "replace_notes_within_box")
# crop model strings to before "_"
df["model"] = df["model"].str.split("_").str[0]

models = df.model.unique()
tasks = df.task.unique()

# Truncate model names to first 6 characters
# df["model"] = df["model"].str[:30]

# Get all metrics columns
metric_cols = [col for col in df.columns if col.startswith("metric/")]

#%%
# Create scatter plots comparing ground truth vs generated samples
for task in [t for t in tasks if t != "ground_truth"]:
    task_df = df[df["task"] == task].copy()
    ground_truth_df = df[df["task"] == "ground_truth"].copy()
    
    # Get dimensions for subplot grid
    n_metrics = len(metric_cols)
    n_models = len(task_df["model"].unique())
    
    # Create figure with subplots for each metric and model
    fig, axes = plt.subplots(n_metrics, n_models, 
                            figsize=(5 * n_models, 4 * n_metrics),
                            squeeze=False)
    fig.suptitle(f"Ground Truth vs Generated Metrics for Task: {task}", 
                size=16, y=1.02)
    
    # Create scatter plot for each metric and model combination
    for metric_idx, metric in enumerate(metric_cols):
        ground_truth_values = ground_truth_df[metric].values
        
        # Find metric-specific min and max for shared axes
        metric_values = [*ground_truth_values]
        for model_values in task_df.groupby("model")[metric].agg(list):
            metric_values.extend(model_values)
        metric_min, metric_max = min(metric_values), max(metric_values)
        
        for model_idx, model in enumerate(task_df["model"].unique()):
            ax = axes[metric_idx, model_idx]
            model_values = task_df[task_df["model"] == model][metric].values
            
            # Create scatter plot
            ax.scatter(ground_truth_values, model_values, alpha=0.6)
            
            # Add diagonal reference line
            ax.plot([metric_min, metric_max], [metric_min, metric_max], 
                   'k--', alpha=0.5)
            
            # Set axes properties
            ax.set_xlim(metric_min, metric_max)
            ax.set_ylim(metric_min, metric_max)
            ax.set_aspect('equal')
            
            # Add labels
            metric_name = metric.replace("metric/", "").replace("_", " ").title()
            if metric_idx == 0:
                ax.set_title(f"Model: {model}")
            if model_idx == 0:
                ax.set_ylabel(f"{metric_name}\nGenerated Value")
            if metric_idx == n_metrics - 1:
                ax.set_xlabel("Ground Truth Value")
    
    plt.tight_layout()
    plt.show()

#%%
# Create violin plots for each task (excluding ground_truth task)
for task in [t for t in tasks if t != "ground_truth"]:
    # Filter data for current task and add ground truth data
    task_df = df[df["task"] == task].copy()
    ground_truth_df = df[df["task"] == "ground_truth"].copy()
    ground_truth_df["task"] = task  # Set task to match for plotting
    combined_df = pd.concat([task_df, ground_truth_df])

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
            # Create violin plot
            sns.violinplot(data=combined_df, x="model", y=metric, ax=axes[idx])

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

#%%
# Calculate summary statistics (including ground truth)
summary_stats = []
for task in [t for t in tasks if t != "ground_truth"]:
    task_df = df[df["task"] == task]
    ground_truth_df = df[df["task"] == "ground_truth"]

    # Regular models
    for model in task_df["model"].unique():
        model_df = task_df[task_df["model"] == model]
        stats = {"task": task, "model": model}
        for metric in metric_cols:
            stats[f"{metric}_mean"] = model_df[metric].mean()
            stats[f"{metric}_std"] = model_df[metric].std()
        summary_stats.append(stats)

    # Ground truth
    stats = {"task": task, "model": "ground"}
    for metric in metric_cols:
        stats[f"{metric}_mean"] = ground_truth_df[metric].mean()
        stats[f"{metric}_std"] = ground_truth_df[metric].std()
    summary_stats.append(stats)

# Convert summary statistics to DataFrame
summary_df = pd.DataFrame(summary_stats)

# Print summary table
print("\nSummary Statistics:")
print(summary_df.to_string())
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

n_cols = 1 + len(models)  # ground truth + each model
n_rows = 5
figsize = (20, 4 * n_examples)  # Adjust height based on number of examples
tpq = 96

# Create visualization for each task
for task in tasks:
    print(f"\nViewing task: {task}")

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot ground truth in first column
    for row, sm in enumerate(ground_truth_midis):
        ax = axes[row, 0]
        roll = piano_roll(sm, tpq)
        im = ax.imshow(roll, aspect="auto", origin="lower", cmap="Blues")
        if row == 0:  # Add column title for ground truth
            ax.set_title("Ground Truth", pad=10)
        ax.set_xlabel("Time (ticks)")
        ax.set_ylabel("MIDI Note")

    # Plot each model's output in subsequent columns
    for col, model in enumerate(models, start=1):
        model_path = base_path / model / task
        model_files = list(model_path.glob("*.mid"))
        model_midis = [symusic.Score(f) for f in model_files]

        for row, sm in enumerate(model_midis):
            ax = axes[row, col]
            roll = piano_roll(sm, tpq)
            im = ax.imshow(roll, aspect="auto", origin="lower", cmap="Blues")
            if row == 0:  # Add column title for each model
                ax.set_title(model, pad=10)
            ax.set_xlabel("Time (ticks)")
            if col > 0:  # Remove redundant y-labels
                ax.set_ylabel("")
                ax.set_yticks([])

    plt.suptitle(f"Task: {task}", size=16, y=1.02)
    plt.tight_layout()
    plt.show()

# %%
