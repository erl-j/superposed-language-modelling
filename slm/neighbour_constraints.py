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
from train import TrainingWrapper
from util import preview_sm, sm_fix_overlap_notes, loop_sm
from tokenizer import instrument_class_to_selected_program_nr
import util
from PAPER_CHECKPOINTS import CHECKPOINTS
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
import numpy as np
from piano_heatmap import create_piano_visualization

USE_FP16 = False

DEVICE = "cuda:7"

model = TrainingWrapper.load_from_checkpoint(
    CHECKPOINTS["slm_mixed_150epochs"],
    map_location=DEVICE,
)

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

def preview(sm):
    sm = sm.copy()
    sm = sm_fix_overlap_notes(sm)
    preview_sm(loop_sm(sm, 4, 4))

if USE_FP16:
    model = model.convert_to_half()

# create 128 bpm rock loop with drums, bass, guitar with max 280 notes
N_EVENTS = model.tokenizer.config["max_notes"]
N_CONSTRAINED = 299 # For these experiments, we'll constrain just one event
N_UNCONSTRAINED = 1  # And observe one unconstrained event

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
def neighbor_constraints(e, ec, n_events):
    e = []
    # force N_CONSTRAINED bass notes
    e += [ec()
          .intersect({"instrument": {"Bass"}})
          .force_active() for _ in range(N_CONSTRAINED)]
    # add N_UNCONSTRAINED inactive events
    e += [ec().force_inactive() for _ in range(N_UNCONSTRAINED)]
    # pad with active events
    e += [ec().force_active() for _ in range(n_events - len(e))]
    return e

e = neighbor_constraints([], ec, N_EVENTS)
mask = model.tokenizer.event_constraints_to_mask(e).to(DEVICE)
out_logits = model.model.forward(mask.float())

# plot distribution 

import matplotlib.pyplot as plt
for attribute in ["pitch", "instrument", "tag"]:
    # get last event logits
    last_event_logits = out_logits[0,-1]

    # softmax
    last_event_probs = torch.softmax(last_event_logits, dim=-1)

    attribute_vocab = [token for token in model.tokenizer.vocab if attribute+":" in token]
    attribute_index = model.tokenizer.note_attribute_order.index(attribute)
    attribute_token_indices = [model.tokenizer.vocab.index(token) for token in attribute_vocab]
    # plot probs for each attribute
    attr_probs = last_event_probs[attribute_index, attribute_token_indices].cpu().detach().numpy()
    # plot bar plot with token names

    # make figure wide globally with rc params
    plt.rcParams['figure.figsize'] = [32, 8]

    plt.figure()
    plt.bar(attribute_vocab, attr_probs)
    plt.title(attribute)
    plt.xticks(rotation=90)
    plt.show()

    # first event logits
    first_event_logits = out_logits[0,0]
    first_event_probs = torch.softmax(first_event_logits, dim=-1)
    attr_probs = first_event_probs[attribute_index, attribute_token_indices].cpu().detach().numpy()

    # plot bar plot with token names
    plt.figure()
    plt.bar(attribute_vocab, attr_probs)
    plt.title(attribute)
    plt.xticks(rotation=90)
    plt.show()

os.makedirs("neighbour_constraints", exist_ok=True)
# save figure
plt.savefig("neighbour_constraints/neighbour_constraints.png")
# %%

def no_constraint():
    e = []
    e += [ec()
          .force_active()]
    e += [ec().force_active()]
    e += [ec().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e

def run_scale_experiment():
    """Test how C and F pitch constraint on event A affects pitch of event B"""
    e = []
    # Event A: Constrained to C and F notes
    e += [ec()
          .intersect({"pitch": {"48", "53"}})  # C4 (60) and F4 (65)
          .force_active()]
    
    # Event B: Unconstrained
    e += [ec().force_active()]
    
    # Pad remaining events
    e += [ec().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e

def run_instrument_experiment():
    """Test how instrument constraint (strings/pipe) affects pitch of same event"""
    e = []
    # Event A: Constrained to strings/pipe
    e += [ec()
          .intersect({"instrument": {"Strings", "Pipe"}})
          .force_active()]
    
    # Event B: Unconstrained
    e += [ec().force_active()]
    
    # Pad remaining events
    e += [ec().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e

def run_pitch_experiment(low_range=True):
    """Test how pitch range affects instrument choice of next event"""
    e = []
    # Event A: Constrained to specific pitch range
    if low_range:
        e += [ec()
              .intersect({"pitch": {str(p) for p in range(36, 47)}})
              .force_active()]
    else:
        e += [ec()
              .intersect({"pitch": {str(p) for p in range(64, 77)}})
              .force_active()]
    
    # Event B: Unconstrained
    e += [ec().force_active()]
    
    # Pad remaining events
    e += [ec().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e


# Run experiments
experiments = {
    "scale_constraint": run_scale_experiment(),
    "instrument_constraint": run_instrument_experiment(),
}

# Plot results for each experiment
for exp_name, e in experiments.items():
    print(f"\nRunning experiment: {exp_name}")
    mask = model.tokenizer.event_constraints_to_mask(e).to(DEVICE)
    out_logits = model.model.forward(mask.float())
    
    # Get event A (constrained) probabilities
    event_a_logits = out_logits[0, 0]  # Index 0 for event A
    event_a_probs = torch.softmax(event_a_logits, dim=-1)
    
    # Get event B (unconstrained) probabilities
    event_b_logits = out_logits[0, 1]  # Index 1 for event B
    event_b_probs = torch.softmax(event_b_logits, dim=-1)
    
    # Get pitch probabilities
    pitch_vocab = [token for token in model.tokenizer.vocab if "pitch:" in token and "-" not in token and "Drums" not in token]
    pitch_index = model.tokenizer.note_attribute_order.index("pitch")
    pitch_token_indices = [model.tokenizer.vocab.index(token) for token in pitch_vocab]
    
    # Process Event A probabilities
    pitch_probs_a = event_a_probs[pitch_index, pitch_token_indices].cpu().detach().numpy()
    keyboard_a = np.zeros((7, 12))
    for i, (token, prob) in enumerate(zip(pitch_vocab, pitch_probs_a)):
        pitch = int(token.split(':')[1])
        octave = (pitch - 21) // 12  # A0 is 21
        note = (pitch - 21) % 12
        if 0 <= octave < 7:  # Only show 7 octaves
            keyboard_a[octave, note] = prob
    
    # Process Event B probabilities
    pitch_probs_b = event_b_probs[pitch_index, pitch_token_indices].cpu().detach().numpy()
    keyboard_b = np.zeros((7, 12))
    for i, (token, prob) in enumerate(zip(pitch_vocab, pitch_probs_b)):
        pitch = int(token.split(':')[1])
        octave = (pitch - 21) // 12  # A0 is 21
        note = (pitch - 21) % 12
        if 0 <= octave < 7:  # Only show 7 octaves
            keyboard_b[octave, note] = prob
    
    # Add constraint information for Event A
    constraint_text = ""
    if "scale" in exp_name:
        constraint_text = "Constraint: C and F notes"
    elif "instrument" in exp_name:
        constraint_text = "Constraint: Strings/Pipe"
    elif "pitch_low" in exp_name:
        constraint_text = "Constraint: Low pitch range (36-46)"
    elif "pitch_high" in exp_name:
        constraint_text = "Constraint: High pitch range (64-76)"
    
    # Create and save the visualizations for both events
    fig_a = create_piano_visualization(keyboard_a, f"Event A (Constrained) - {constraint_text}")
    save_path_a = os.path.join("neighbour_constraints", f"{exp_name}_event_a_visualization.png")
    print(f"Saving Event A plot to: {save_path_a}")
    fig_a.savefig(save_path_a, dpi=300, bbox_inches='tight')
    plt.close(fig_a)
    
    fig_b = create_piano_visualization(keyboard_b, f"Event B (Unconstrained) - After {constraint_text}")
    save_path_b = os.path.join("neighbour_constraints", f"{exp_name}_event_b_visualization.png")
    print(f"Saving Event B plot to: {save_path_b}")
    fig_b.savefig(save_path_b, dpi=300, bbox_inches='tight')
    plt.close(fig_b)

print("\nAll experiments completed. Check the neighbour_constraints directory for the plots.")
