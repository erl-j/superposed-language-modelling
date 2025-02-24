#%%
import sys
import socket
import os
import random
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
import symusic
sys.path.append("slm/")
# from slm.train_old import EncoderOnlyModel
from train import TrainingWrapper
from util import preview_sm, sm_fix_overlap_notes, loop_sm
from tokenizer import instrument_class_to_selected_program_nr
import util
from paper_checkpoints import CHECKPOINTS
from constraints.addx import *
from constraints.re import *
from constraints.templates import *
from constraints.core import (
    MusicalEventConstraint,
    DRUM_PITCHES,
    PERCUSSION_PITCHES,
    TOM_PITCHES,
    CRASH_PITCHES,
    HIHAT_PITCHES,
)
from transformers import pipeline
import pretty_midi
import time
from conversion_utils import looprep_to_sm, sm_to_events, sm_to_looprep
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
    # sm_filter_fn = lambda sm: not any(
    #     track.program == 0 and not track.is_drum and "piano" not in track.name.lower()
    #     for track in sm.tracks
    # )

    tempo_filter_fn = lambda sm: len(sm.tempos) > 0

    test_ds = MidiDataset(
        cache_path="../data/gmd_loops/tst_midi_records_loops.pt",
        n_bars = 4,
        path_filter_fn=mmd_4bar_filter_fn,
        genre_list=tokenizer.config["tags"],
        tokenizer=tokenizer,
        min_notes=16,
        max_notes=tokenizer.config["max_notes"],
        use_random_shift=False,
        sm_filter_fn=tempo_filter_fn,
        use_midi_type=False,
    )
    return test_ds


# Set device
DEVICE = f"cuda:3"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the test dataset using first available model's tokenizer
print("\nInitializing with first available model...")
first_model = setup_model(next(iter(CHECKPOINTS.values())), DEVICE)
test_dataset = load_test_dataset(first_model.tokenizer)

#%%
# load the dataset
samples = []
for i in range(N_EXAMPLES):
    samples.append(test_dataset[i]["token_ids"])
samples = torch.stack(samples).to(DEVICE)

#%%

# one hot encode
one_hot = torch.nn.functional.one_hot(samples, num_classes=len(first_model.tokenizer.vocab)).float()

# %%
# one hot encode
print(one_hot.shape)

#%%

model = first_model
# create 128 bpm rock loop with drums, bass, guitar with max 280 notes
N_EVENTS = model.tokenizer.config["max_notes"]

def generate(
    mask,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    tokens_per_step=1,
    attribute_temperature=None,
    order="random",
):

    out = model.generate(
        mask,
        temperature=temperature,
        tokens_per_step=tokens_per_step,
        top_p=top_p,
        top_k=top_k,
        order=order,
        attribute_temperature=attribute_temperature,
    )[0].argmax(-1)
    
    return out

def generate_from_constraints(e, sampling_params={}):
    mask = model.tokenizer.event_constraints_to_mask(e).to(DEVICE)
    x = generate(
        mask,
        temperature=sampling_params.get("temperature", 1.0),
        top_p=sampling_params.get("top_p", 1.0),
        top_k=sampling_params.get("top_k", 0),
        tokens_per_step=sampling_params.get("tokens_per_step", 1),
        attribute_temperature=sampling_params.get("attribute_temperature", None),
        order=sampling_params.get("order", "random"),
    )
    x_sm = model.tokenizer.decode(x)
    x_sm = util.sm_fix_overlap_notes(x_sm)
    return x_sm

ec = lambda: MusicalEventConstraint(model.tokenizer)

e = [ ec().intersect({"instrument": {"Piano", "-"}}) for _ in range(N_EVENTS) ]

mask = model.tokenizer.event_constraints_to_mask(e).to(DEVICE)
print(mask.shape)
# %%
# filter data by constraint. 
follows_constraint = (one_hot * mask).sum(dim=-1) > 0
# check that it is true across event and attribute dimensions
follows_constraint = follows_constraint.all(dim=[1, 2])
print(follows_constraint.shape)

print(follows_constraint.sum())

print(follows_constraint)
# %%
