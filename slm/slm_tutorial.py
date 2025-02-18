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
    e += [ec().intersect({"instrument": {"Bass"}}).force_active() for _ in range(3)]
    # force 1 drums note
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for _ in range(3)]

    # constrain instrument to be only bass and drums
    e += [ec().intersect({"instrument": {"Bass", "Drums"}}).force_active() for i in range(50)]

    # add 50 optional bass and drums
    e += [ec().intersect({"instrument": {"Bass", "Drums", "-"}}) for i in range(50)]
    # pad
    # set tag to pop
    e = [ev.intersect({"tag": {"rock", "-"}}) for ev in e]
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

e = bass_and_drums([], ec, N_EVENTS)
preview_sm(generate_from_constraints(e))
# %%
# now we'll create a pentatonic chromatic percussion loop

def chromatic_percussion(e, ec, n_events):

    e = [
        ec().intersect({"instrument": {"Chromatic Percussion"}, "duration": {"1/2"}})
        .intersect(ec().pitch_in_scale_constraint("C pentatonic", (60, 108)))
        .force_active() for i in range(50)
    ]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    return e

e = chromatic_percussion([], ec, N_EVENTS)
preview_sm(generate_from_constraints(e))

#%%
# now we'll create a drum loop with lots of cowbells
def cowbell_drum_loop(e, ec, n_events):
    e = [
        ec().intersect({"instrument": {"Drums"}, "pitch": {"56"}}).force_active() for i in range(10)
    ]
    # add more drums
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for i in range(40)]
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

e = cowbell_drum_loop([], ec, N_EVENTS)
preview_sm(generate_from_constraints(e))
# %%
# now we'll create a fast guitar arpeggio loop

def fast_guitar_arpeggio(e, ec, n_events):

    # one guitar note every 16th note

    e = [
        ec().intersect({"instrument": {"Guitar"}, "duration": {"1/16"}, "onset/global_tick": {str(onset_tick)}})
        .intersect(ec().pitch_in_scale_constraint("C pentatonic", (48, 84)))
        .force_active() for onset_tick in range(0, 24*16, 6)
    ]

    # set tempo to 120 bpm
    e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    return e

e = fast_guitar_arpeggio([], ec, N_EVENTS)
preview_sm(generate_from_constraints(e,{"attribute_temperature": {"pitch": 1.5}}))

# we can increase the temperature of pitch 

#%%
# now a drum beat with triplets

def drum_triplets(e, ec, n_events):
    # one drum note every 12th note

    e = [
        ec().intersect({"instrument": {"Drums"}, "onset/global_tick": {str(onset_tick)}}).force_active() for onset_tick in range(0, 24*12, 8)
    ]

    # add 32 more drums
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for i in range(32)]

    # add bass 10 bass notes on any triplet
    triplet_ticks = {str(t) for t in range(0, 24*12, 8)}

    e += [
        ec().intersect({"instrument": {"Bass"}, "onset/global_tick": triplet_ticks}).force_active() for i in range(12)
    ]

    # set tempo to 120 bpm
    e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    return e

e = drum_triplets([], ec, N_EVENTS)

preview_sm(generate_from_constraints(e))

# %%

# now a piece with piano chords and flute
def pads(e, ec, n_events):

    e = [
        ec().intersect({"instrument": {"Synth Pad"}, "duration": {"1/2"}}).force_active() for i in range(64)
    ]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    return e

e = pads([], ec, N_EVENTS)
preview_sm(generate_from_constraints(e, {"temperature": 0.95}))

# %%

# we want to enforce the chord progression C - G - Am - F

def chord_progression(e, ec, n_events):

    # create 4 sets, one for each chord

    chord_to_pitches = {
        "C": [60, 67, 72, 76, 79, 84, 88, 91, 96, 100, 103],
        "G": [55, 62, 67, 71, 74, 79, 83, 86, 91, 95, 98],
        "Am": [57,  64, 69, 72, 76, 81, 84, 88, 93, 96, 100],
        "F": [53, 60, 65, 69, 72, 77, 81, 84, 89, 93, 96],
    }

    chord_progression = [{"start_beat": 0, 
                          "end_beat": 4,
                          "chord": "C"}, 
                          {"start_beat": 4, 
                            "end_beat": 8,
                           "chord": "G"}, 
                           {"start_beat": 8, 
                            "end_beat": 12,
                            "chord": "Am"}, 
                            {"start_beat": 12,
                            "end_beat": 16,
                            "chord": "Am"}
                        ]
    e = []
    for chord_idx in range(len(chord_progression)):
        tick_range = { str(s) for s in range(int(chord_progression[chord_idx]["start_beat"]*24), int(chord_progression[chord_idx]["end_beat"]*24))}
        e += [
            ec().intersect({"instrument": {"Piano"}, 
                            "pitch": {str(s) for s in chord_to_pitches[chord_progression[chord_idx]["chord"]]}
                            ,"onset/global_tick": tick_range,
                            "offset/global_tick": tick_range})
            for i in range(16)
        ]

    # add 40 flute notes
    e += [
        ec().intersect({"instrument": {"Guitar"}, "duration": {"1/4", "1/8"}}).force_active() for i in range(40)
    ]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    return e

e = chord_progression([], ec, N_EVENTS)

preview_sm(generate_from_constraints(e, {"temperature": 1.0}))





# %%
