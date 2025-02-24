#%%
import os
import torch
import random
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import time
from train import TrainingWrapper
from data import MidiDataset
from conversion_utils import sm_to_events
from constraints.core import MusicalEventConstraint
from paper_checkpoints import CHECKPOINTS
from util import sm_set_track_order, sm_fix_overlap_notes

# Number of examples to generate per task
N_EXAMPLES = 250
GENERATE = True
ORDER = "random"

OUTPUT_DIR = Path("./artefacts/applications_250e")


def setup_model(checkpoint_path, device):
    """Load and set up the model."""
    print(f"Loading model from {checkpoint_path}...")
    model = TrainingWrapper.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model

def load_test_dataset(tokenizer):
    """Load the test dataset."""
    print("Loading test dataset...")
    mmd_4bar_filter_fn = lambda x: "n_bars=4" in x
    sm_filter_fn = lambda sm: not any(
        track.program == 0 and not track.is_drum and "piano" not in track.name.lower()
        for track in sm.tracks
    )

    test_ds = MidiDataset(
        cache_path="../data/mmd_loops/tst_midi_records_unique_pr.pt",
        n_bars = 4,
        path_filter_fn=mmd_4bar_filter_fn,
        genre_list=tokenizer.config["tags"],
        tokenizer=tokenizer,
        min_notes=16,
        max_notes=tokenizer.config["max_notes"],
        use_random_shift=False,
        sm_filter_fn=sm_filter_fn,
        use_midi_type=False,
    )
    return test_ds


# Set device
device = f"cuda:3"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the test dataset using first available model's tokenizer
print("\nInitializing with first available model...")
first_model = setup_model(next(iter(CHECKPOINTS.values())), device)
test_dataset = load_test_dataset(first_model.tokenizer)

#%%
# load the dataset
samples = []
for i in range(N_EXAMPLES):
    samples.append(test_dataset[i]["token_ids"])
samples = torch.stack(samples).to(device)


# %%
