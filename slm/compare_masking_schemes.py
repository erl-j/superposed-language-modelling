#%%
import torch
from slm.tokenizer import Tokenizer
from data import MidiDataset
from fractions import Fraction
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from slm.train2_old import (
    random_add_masking_mml,
    random_add_masking_variable_superposition,
    random_add_masking_variable_superposition_ratio,
)

# Dataset configuration
tokenizer_config = {
    "ticks_per_beat": 24,
    "time_hierarchy": "tick",
    "pitch_range": [0, 128],
    "max_beats": 16,
    "max_notes": 300,
    "min_tempo": 40,
    "max_tempo": 300,
    "n_tempo_bins": 32,
    "n_velocity_bins": 32,
    "time_signatures": None,
    "tags": open("../data/mmd_loops/tags.txt").read().splitlines(),
    "shuffle_notes": True,
    "use_offset": True,
    "merge_pitch_and_beat": False,
    "use_program": False,
    "use_instrument": True,
    "ignored_track_names": [f"Layers{i}" for i in range(0, 8)],
    "separate_drum_pitch": True,
    "use_drum_duration": False,
    "use_durations": True,
    "durations": [
        Fraction(1, 32),
        Fraction(1, 16),
        Fraction(1, 8),
        Fraction(1, 4),
        Fraction(1, 2),
        Fraction(1, 1),
        Fraction(2, 1),
        Fraction(4, 1),
    ],
}

tokenizer = Tokenizer(tokenizer_config)

# Initialize dataset
val_ds = MidiDataset(
    cache_path="../data/mmd_loops/val_midi_records_unique_pr.pt",
    path_filter_fn=lambda x: "n_bars=4" in x,
    genre_list=tokenizer_config["tags"],
    tokenizer=tokenizer,
    min_notes=16,
    max_notes=tokenizer_config["max_notes"],
    use_random_shift=False,
)

format_mask = torch.Tensor(tokenizer.get_format_mask())

# Batch processing
n_samples = 200
batch_size = 100
n_batches = (n_samples + batch_size - 1) // batch_size

all_masked_mml = []
all_masked_var_super = []
all_masked_var_super_ratio = []
all_originals = []

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, n_samples)
    batch_samples = [val_ds[j] for j in range(start_idx, end_idx)]

    # Stack sequences
    sequences = torch.stack([sample["token_ids"] for sample in batch_samples])
    one_hot = F.one_hot(sequences, len(tokenizer.vocab)).float()

    # Apply masking schemes in batch
    masked_mml = random_add_masking_mml(one_hot) * format_mask[None, ...]
    masked_var_super = (
        random_add_masking_variable_superposition(one_hot) * format_mask[None, ...]
    )
    masked_var_super_ratio = (
        random_add_masking_variable_superposition_ratio(one_hot, format_mask)
        * format_mask[None, ...]
    )

    all_masked_mml.append(masked_mml)
    all_masked_var_super.append(masked_var_super)
    all_masked_var_super_ratio.append(masked_var_super_ratio)
    all_originals.append(one_hot)

# Concatenate results
all_masked_mml = torch.cat(all_masked_mml, dim=0)
all_masked_var_super = torch.cat(all_masked_var_super, dim=0)
all_masked_var_super_ratio = torch.cat(all_masked_var_super_ratio, dim=0)
all_originals = torch.cat(all_originals, dim=0)


def calculate_detailed_stats(masked_tensor):
    vocab_sums = masked_tensor.sum(dim=-1)
    undetermined_positions = (vocab_sums > 1).float().mean().item()
    determined_positions = (vocab_sums == 1).float().mean().item()
    ones_per_column = masked_tensor.sum().item() / (
        masked_tensor.shape[0] * masked_tensor.shape[1]
    )
    return undetermined_positions, determined_positions, ones_per_column


# Calculate and print statistics
print("\nDetailed Statistics:")
for name, masked in [
    ("MML", all_masked_mml),
    ("Variable Superposition", all_masked_var_super),
    ("Variable Superposition Ratio", all_masked_var_super_ratio),
]:
    undetermined, determined, ones_avg = calculate_detailed_stats(masked)
    print(f"\n{name}:")
    print(f"Average undetermined positions: {undetermined:.4f}")
    print(f"Average determined positions: {determined:.4f}")
    print(f"Average ones per column: {ones_avg:.4f}")

# Visualization
plt.figure(figsize=(20, 10))
examples_to_show = 5
fig, axes = plt.subplots(4, examples_to_show, figsize=(20, 8))
plt.set_cmap("binary")

for i in range(examples_to_show):
    axes[0, i].imshow(all_originals[i].T, interpolation="none")
    axes[1, i].imshow(all_masked_mml[i].T, interpolation="none")
    axes[2, i].imshow(all_masked_var_super[i].T, interpolation="none")
    axes[3, i].imshow(all_masked_var_super_ratio[i].T, interpolation="none")

    for ax in axes[:, i]:
        ax.set_xticks([])
        ax.set_yticks([])

row_labels = [
    "Original",
    "MML Masking",
    "Variable Superposition",
    "Variable Superposition Ratio",
]

for ax, label in zip(axes[:, 0], row_labels):
    ax.set_ylabel(label, rotation=90, size="large")

plt.tight_layout()
plt.show()

# %%
