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
        cache_path="../data/gmd_loops_2/val_midi_records_loops.pt",
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

#%%
# len of test_ds

# Set device
DEVICE = f"cuda:3"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the test dataset using first available model's tokenizer
print("\nInitializing with first available model...")
first_model = setup_model(next(iter(CHECKPOINTS.values())), DEVICE)
test_dataset = load_test_dataset(first_model.tokenizer)

#%% look at len of test_ds

#%%

dl = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

#%%

import matplotlib.pyplot as plt

import numpy as np
import pretty_midi
from util import crop_sm
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
from util import crop_sm

def detail_plot(sm):
    pr_tpq = 12
    tempo = int(sm.tempos[-1].qpm)
    time_sig = f"{sm.time_signatures[-1].numerator}/{sm.time_signatures[-1].denominator}"
    
    # Get instrument names
    instrument_names = [pretty_midi.program_to_instrument_name(track.program) if not track.is_drum else "Drums" for track in sm.tracks]
    
    # Get pianoroll data
    pr = sm.copy().resample(pr_tpq, min_dur=0).pianoroll(modes=["frame"])[0]
    
    # Get unique instruments
    unique_instruments = np.unique(instrument_names)

    # Create a mapping from instrument names to list of track indices
    instrument_to_track_indices = {instrument: [] for instrument in unique_instruments}
    for i, instrument in enumerate(instrument_names):
        instrument_to_track_indices[instrument].append(i)
    
    # We know that it's 4 bars so let's crop it
    loop_ticks = pr_tpq * 4 * 4
    
    # Use a colormap with distinguishable colors
    colors = plt.cm.tab10.colors

    # Get drum track indices
    drum_indices = np.where(np.array(instrument_names) == "Drums")[0]
    has_drums = len(drum_indices) > 0
    
    # Get melodic track indices (all non-drum tracks)
    melodic_indices = [i for i, name in enumerate(instrument_names) if name != "Drums"]
    
    # Get all non-zero pitches from melodic instruments to determine the y-axis range
    all_melodic_pitches = []
    for idx in melodic_indices:
        instrument_pr = pr[idx][:, :loop_ticks]
        pitches = np.where(np.any(instrument_pr > 0, axis=1))[0]
        all_melodic_pitches.extend(pitches)
    
    if all_melodic_pitches:
        # Find the min and max pitches used with padding of ±3
        min_pitch = max(0, min(all_melodic_pitches) - 3)
        max_pitch = min(127, max(all_melodic_pitches) + 3)
    else:
        # Default range if no melodic notes found
        min_pitch = 60 - 24
        max_pitch = 60 + 24
    
    # Set up the figure with two subplots if there are drums
    if has_drums:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(hspace=0.3)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot melodic instruments on the top subplot
    legend_handles = []
    for i, instrument_name in enumerate(unique_instruments):
        if instrument_name == "Drums":
            continue
        
        # Get indices of tracks with this instrument_name
        track_indices = instrument_to_track_indices[instrument_name]
        
        # Sum all channels with the same program number
        instrument_pr = pr[track_indices].sum(axis=0)
        instrument_pr = instrument_pr[:, :loop_ticks]
        
        # Only show non-zero pitches
        non_zero_indices = np.where(instrument_pr > 0)
        
        # Plot with distinct color
        color = colors[i % len(colors)]
        scatter = ax1.scatter(non_zero_indices[1], non_zero_indices[0], 
                             color=color, marker='s', s=10, 
                             label=instrument_name)
        legend_handles.append(scatter)
    
    # Calculate beat positions
    lines_tpq = 1
    
    # Create tick positions for all data points
    time_ticks = np.arange(0, loop_ticks, lines_tpq * pr_tpq)
    
    # Create matching labels
    time_labels = np.arange(0, len(time_ticks))
    
    # Remove default grid
    ax1.grid(False)

    # Add only vertical lines for each beat
    for tick in time_ticks:
        ax1.axvline(x=tick, color='gray', linestyle='--', alpha=0.3)
    
    # Add horizontal lines for each octave
    for octave in range(min_pitch // 12, (max_pitch // 12) + 1):
        pitch = octave * 12
        if min_pitch <= pitch <= max_pitch:
            ax1.axhline(y=pitch, color='gray', linestyle='-', alpha=0.2)
    
    ax1.set_xticks(time_ticks)
    ax1.set_xticklabels(time_labels)
    
    # Set the x and y limits to focus on the actual used pitch range
    ax1.set_xlim(0, loop_ticks)
    ax1.set_ylim(min_pitch, max_pitch)
    
    # Add pitch labels with note names every 12 semitones (C notes)
    pitch_ticks = [pitch for pitch in range(min_pitch - (min_pitch % 12), max_pitch + 12, 12) if min_pitch <= pitch <= max_pitch]
    pitch_labels = [f"C{(pitch // 12) - 1}" for pitch in pitch_ticks]  # C4 is MIDI 60
    ax1.set_yticks(pitch_ticks)
    ax1.set_yticklabels(pitch_labels)
    
    ax1.set_xlabel("Beat")
    ax1.set_ylabel("Pitch")
    ax1.set_title(f"Melodic Instruments - Tempo: {tempo}, Time Signature: {time_sig}")
    
    # Add legend
    ax1.legend(handles=legend_handles, title="Instruments", 
               loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Plot drums if they exist
    if has_drums:
        # Create mapping for drum pitch to name
        pitch_to_drum_name = {
            pitch: pretty_midi.note_number_to_drum_name(pitch) 
            for pitch in range(0, 128)
            if pretty_midi.note_number_to_drum_name(pitch) is not None
        }
        
        # Get drum pianoroll
        drum_pr = np.zeros_like(pr[0])
        for idx in drum_indices:
            drum_pr += pr[idx]
            
        drum_pr = drum_pr[:, :loop_ticks]
        
        # Get non-zero indices in the drum track
        drum_non_zero = np.where(drum_pr > 0)
        
        # Get unique drum pitches used
        unique_drum_pitches = np.unique(drum_non_zero[0])
        
        # Create a mapping of actual pitch to display position
        pitch_to_position = {pitch: i for i, pitch in enumerate(unique_drum_pitches)}
        
        # Map the pitches to display positions
        display_positions = [pitch_to_position[pitch] for pitch in drum_non_zero[0]]
        
        # Plot drum hits
        ax2.scatter(drum_non_zero[1], display_positions, 
                   color='black', marker='s', s=20)
        
        # Add only vertical lines for each beat (same as above)
        for tick in time_ticks:
            ax2.axvline(x=tick, color='gray', linestyle='--', alpha=0.3)
        
        # Set x-axis ticks and limits to match the main plot
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(time_labels)
        ax2.set_xlim(0, loop_ticks)
        
        # Set y-axis labels with drum names
        ax2.set_yticks(range(len(unique_drum_pitches)))
        drum_labels = [f"{pitch} - {pitch_to_drum_name.get(pitch, 'Unknown')}" 
                      for pitch in unique_drum_pitches]
        ax2.set_yticklabels(drum_labels)
        
        # Set y limits with a bit of padding
        ax2.set_ylim(-0.5, len(unique_drum_pitches) - 0.5)
        
        ax2.set_xlabel("Beat")
        ax2.set_ylabel("Drum Type")
        ax2.set_title("Drum Pattern")
    
    plt.tight_layout()
    plt.show()

loops_in_parent_song = []
# preview 10 samples
for i in range(1):
    batch = next(iter(dl))
    print(f"n loops in parent song: {batch['n_loops_in_parent_song'].item()}")
    token_ids = batch["token_ids"].squeeze().to(DEVICE)
    print(token_ids.shape)
    print(first_model.tokenizer.indices_to_tokens(token_ids[0]))
    sm = first_model.tokenizer.decode(token_ids)
    preview_sm(sm)
    detail_plot(sm)

# print(loops_in_parent_song)

#%%


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

#%%

tpq = model.tokenizer.config["ticks_per_beat"]
n_beats = 16
constraints = {
    "piano" : [ ec().intersect({"instrument": {"Piano", "-"}}) for _ in range(N_EVENTS) ],
    "drum_and_bass_and_guitar" : [ ec().intersect({"instrument": {"Drums", "Bass", "Guitar","-"}}) for _ in range(N_EVENTS) ],
    "drum_and_bass" : [ ec().intersect({"instrument": {"Drums", "Bass","-"}}) for _ in range(N_EVENTS) ],
    "1/2 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, tpq//2)} | {"-"} }) for _ in range(N_EVENTS) ],
    "1/4 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, tpq//4)} | {"-"} }) for _ in range(N_EVENTS) ],
    "1/8 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, tpq//8)} | {"-"} }) for _ in range(N_EVENTS) ],
    "1/16 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, tpq//16)} | {"-"} }) for _ in range(N_EVENTS) ],
    "c major pitch set" : [ ec().intersect({"pitch": ec().pitch_in_scale_constraint("C major", (10, 127))["pitch"] | {"-"} }) for _ in range(N_EVENTS) ],
    "c pentatonic" : [ ec().intersect({"pitch": ec().pitch_in_scale_constraint("C pentatonic", (10, 127))["pitch"] | {"-"} }) for _ in range(N_EVENTS) ],
}

for constraint_name, e in constraints.items():
    mask = model.tokenizer.event_constraints_to_mask(e).to(DEVICE)
    # filter data by constraint. 
    follows_constraint = (one_hot * mask).sum(dim=-1) > 0
    # check that it is true across event and attribute dimensions
    follows_constraint = follows_constraint.all(dim=[1, 2])

    print(follows_constraint.sum())

# %%
