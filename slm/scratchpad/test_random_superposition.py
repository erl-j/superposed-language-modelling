#%%
import torch
from tokenizer import Tokenizer
from fractions import Fraction
from data import MidiDataset

SEED = 0

torch.manual_seed(SEED)

DATASET = "mmd_loops"

# BATCH_SIZE = 80
# BATCH_SIZE = 40
BATCH_SIZE = 10

tag_list = open(f"../data/{DATASET}/tags.txt").read().splitlines()

N_BARS = 4 if DATASET == "harmonic" else 4
tokenizer_config = {
"ticks_per_beat": 24
if (DATASET == "mmd_loops" or DATASET == "harmonic")
else 48,
"time_hierarchy": "tick",
"pitch_range": [0, 128],
"max_beats": 4 * N_BARS,
"max_notes": 75 * N_BARS if DATASET == "mmd_loops" else 20 * N_BARS,
"min_tempo": 40,
"max_tempo": 300,
"n_tempo_bins": 32,
"n_velocity_bins": 32,
"time_signatures": None,
"tags": tag_list,
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
"fold_event_attributes": False,
}

USE_RANDOM_SHIFT = False
tokenizer = Tokenizer(tokenizer_config)

mmd_4bar_filter_fn = lambda x: f"n_bars={N_BARS}" in x

# if a track has program 0 and is not a drum track and does not contain the word "piano" in the name, filter out the whole midi
# we can't risk having mislabelled tracks in the dataset
sm_filter_fn = lambda sm: not any(
    track.program == 0 and not track.is_drum and "piano" not in track.name.lower()
    for track in sm.tracks
)

val_ds = MidiDataset(
        cache_path=f"../data/{DATASET}/val_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn if DATASET == "mmd_loops" else None,
        genre_list=tag_list,
        tokenizer=tokenizer,
        min_notes=4 * N_BARS if DATASET == "mmd_loops" else 4 * N_BARS,
        max_notes=tokenizer_config["max_notes"],
        use_random_shift=USE_RANDOM_SHIFT,
        sm_filter_fn=sm_filter_fn,
    )

val_dl = torch.utils.data.DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

#%%
from masking import random_superposition

# load one batch with dl
batch = next(iter(val_dl))["token_ids"]

#%%
# one hot encode
one_hot = torch.nn.functional.one_hot(batch, len(tokenizer.vocab)).float()
syntax_mask = torch.Tensor(tokenizer.get_syntax_mask())
# get a random superposition
superposition = random_superposition(one_hot, syntax_mask)

# get some statistics.
# 1. how many elements have more than 1 activation?
print(f"Number of elements with more than 1 activation: {torch.sum(torch.sum(superposition, dim=-1) > 1)}")

# number of elements with exactly 1 activation
print(f"Number of elements with exactly 1 activation: {torch.sum(torch.sum(superposition, dim=-1) == 1)}")
# number of elements with exactly 2 activations per attribute
print(f"Number of elements with exactly 2 activations per attribute: {torch.sum(torch.sum(superposition, dim=-1) == 2)}")
# number of elements with exactly 5 activations per attribute
print(f"Number of elements with exactly 5 activations per attribute: {torch.sum(torch.sum(superposition, dim=-1) == 5)}")

# 2. how many elements have no activation?
print(f"Number of elements with no activation: {torch.sum(torch.sum(superposition, dim=-1) == 0)}")

# make histogram of number of activations per attribute
import matplotlib.pyplot as plt
for i in range(0, len(tokenizer.note_attribute_order)):
    attr_activations = torch.sum(superposition[:, :, i], dim=-1).flatten().numpy()
    plt.hist(attr_activations, alpha=0.5, label=tokenizer.note_attribute_order[i])
    plt.legend()
    plt.show()

# plot first random superposition
plt.imshow(superposition[2,0].T, aspect="auto", interpolation="none")
plt.show()

#%%
# plot total histogram
total_activations = torch.sum(superposition, dim=-1).flatten().numpy()
plt.hist(total_activations, alpha=0.5, label="total activations")
plt.legend()
plt.show()


# %%
from masking import position_mask

# get a position mask
batch_size = 1000
n_positions = 10

position_mask = position_mask(batch_size, n_positions)

print(position_mask.shape)

# print min number of activations
print(f"Min number of activations: {torch.min(torch.sum(position_mask, dim=1))}")

# print max number of activations
print(f"Max number of activations: {torch.max(torch.sum(position_mask, dim=1))}")

# print number of activations per position
print(f"Number of activations per position: {torch.sum(position_mask, dim=1)}")

# plot histogram of number of activations per position
plt.hist(torch.sum(position_mask, dim=1).flatten().numpy())
plt.show()



# %%
