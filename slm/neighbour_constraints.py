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

def bass_and_drums(e, ec, n_events):
    e = []
    # force 1 bass note
    e += [ec()
          #.intersect({"instrument": {"Piano"}})
          .intersect({"instrument": {"Strings", "Ensemble", "Pipe", "Brass"}})
        #   .intersect(ec().pitch_in_scale_constraint("C major", (90,103)))
        #   .intersect({"tag":{"classical"}})
          .force_active() for _ in range(150)]
    # e += [ec().force_active() for _ in range(3)]

    # add 100 inactive events
    e += [ec().force_inactive() for _ in range(100)]

    # pad with active events
    e += [ec().force_active() for _ in range(n_events - len(e))]

    # set last 10 events to be inactive
    return e

e = bass_and_drums([], ec, N_EVENTS)
mask = model.tokenizer.event_constraints_to_mask(e).to(DEVICE)

print(mask.shape)

out_logits = model.model.forward(mask.float())

print(out_logits.shape)

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

    # make figure wide
    plt.figure(figsize=(10,5))
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


# %%
