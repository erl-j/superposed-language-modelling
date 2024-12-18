# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import symusic
from util import piano_roll
import muspy
import tqdm

# Load ground truth examples
base_path = Path("../artefacts/applications")
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
    }
    for record in records
]

print(f"Processed {len(records)} records")
    
#%%
import pandas as pd

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
