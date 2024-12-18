# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import symusic
from util import piano_roll

# %%
# Load ground truth examples
base_path = Path("../artefacts/applications")
ground_truth_path = base_path / "ground_truth"
ground_truth_files = list(ground_truth_path.glob("*.mid"))
ground_truth_midis = [symusic.Score(f) for f in ground_truth_files]

# %%
# Dynamically discover models and tasks
models = [
    d.name for d in base_path.iterdir() if d.is_dir() and d.name != "ground_truth"
]
# Get tasks from first model's directory (assuming all models have same tasks)
first_model_path = base_path / models[0]
tasks = [d.name for d in first_model_path.iterdir() if d.is_dir()]

print(f"Discovered models: {models}")
print(f"Discovered tasks: {tasks}")

# %%
# Configuration
n_examples = len(ground_truth_midis)
n_cols = 1 + len(models)  # ground truth + each model
n_rows = n_examples
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
