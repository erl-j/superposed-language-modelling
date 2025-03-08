#%%
import sys
import socket
import os
import random
import torch
import symusic
sys.path.append("slm/")
# from slm.train_old import EncoderOnlyModel
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

USE_FP16 = False

DEVICE = "cuda:2"

model = TrainingWrapper.load_from_checkpoint(
    # CHECKPOINTS["slm_mixed_150epochs"],
    # "../checkpoints/iconic-firefly-751/last.ckpt",
    # "../checkpoints/playful-wood-749/last.ckpt",
    # "../checkpoints/lucky-wildflower-748/last.ckpt",
    # "../checkpoints/swept-night-752/epoch=25-step=133432-val/loss_epoch=0.12887.ckpt",
    # "../checkpoints/trim-dawn-754/last.ckpt",
    # "../checkpoints/effortless-sun-758/last.ckpt",
    # "../checkpoints/usual-flower-759/last.ckpt",
    # "../checkpoints/devout-terrain-760/last.ckpt",
    # "../checkpoints/azure-darkness-763/last.ckpt",
    # "../checkpoints/snowy-gorge-764/last.ckpt",
    # CHECKPOINTS["slm_mixed_150epochs"],
    # CHECKPOINTS["mlm_150epochs"],
    "../checkpoints/comfy-bush-766/last.ckpt",
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


def sm_to_constraint(sm, tag="other"):
    return sm_to_events(sm, tag, tokenizer=model.tokenizer)

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
    notes_before_removing_overlap = x_sm.note_num()
    x_sm = util.sm_fix_overlap_notes(x_sm)
    notes_after_removing_overlap = x_sm.note_num()
    print(f"Number of overlapping notes: {notes_before_removing_overlap - notes_after_removing_overlap}")
    return x_sm

ec = lambda: MusicalEventConstraint(model.tokenizer)
#%%

# just drums
def piano(e, ec, n_events):
    # add 20 bass notes
    e=[]
    e += [ec().intersect({"instrument": {"Piano", "-"}}).force_active() for _ in range(50)]
    # e += [ec().intersect({"instrument": {"Piano"}}).force_active() for i in range(30)]
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    # set tag to metal
    e = [ev.intersect({"tag": {"classical", "-"}}) for ev in e]
    # set tempo to 120
    e = [ev.intersect(ec().tempo_constraint
                      (140)) for ev in e]
    # restrict to c major
    e = [ev.intersect(ec().pitch_in_scale_constraint("C major", (38, 101))) for ev in e]
    print(model.tokenizer.config["midi_types"])
    if "loop" in model.tokenizer.config["midi_types"]:
        e = [ev.intersect({"midi_type": {"loop", "-"}}) for ev in e]

    return e

e = piano([], ec, N_EVENTS)
sm = generate_from_constraints(e, {"temperature": 1.0})
print(sm.note_num())
preview_sm(loop_sm(sm, 4, 2))

#%% 
def drum_beat_with_syncopation(e, ec, n_events):
    e = []
    # add 32 drums
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for i in range(52)]
    tpq = 24
    n_beats = 16
    # add optional drums on 8th note upbeats, snares or kicks
    e += [ec().intersect({"instrument": {"Drums"},
                          "pitch": {"36 (Drums)"},
                          "onset/global_tick" : {str(t) for t in range(tpq//4, tpq*n_beats, tpq)}}
                          ).force_active() for i in range(8)]                     
    
    # set tempo to 140
    e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
    # pad
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

e = drum_beat_with_syncopation([], ec, N_EVENTS)
sm = generate_from_constraints(e, {"temperature": 1.0})
print(sm.note_num())
preview_sm(loop_sm(sm, 4, 2))

def add_locked_in_bass_line(e, ec, n_events):

    print(f"n_events: {n_events}")
    # remove inactive bass notes
    e = [ev for ev in e if ev.is_active()]
    print(f"n_events: {len(e)}")
    # add bass notes with bass line
    # get drum onsets
    drum_onsets = set()
    for ev in e:
        if "Drums" in ev["instrument"] and "36 (Drums)" in ev["pitch"]:
            drum_onsets.update(ev["onset/global_tick"])

    print(drum_onsets)

    # now add a bass note on every drum onset
    e += [ec().intersect({"instrument": {"Bass"}, 
                          "onset/global_tick": drum_onsets
                          }
                          ).force_active() for i in range(10)]
    
    # pad with inactive notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    return e

e = sm_to_constraint(sm)
e = add_locked_in_bass_line(e, ec, N_EVENTS)
sm = generate_from_constraints(e, {"temperature": 1.0})
print(sm.note_num())
preview_sm(loop_sm(sm, 4, 2))


#%%
def bass_and_drums(e, ec, n_events):
    e = []
    # force 1 bass note
    e += [ec().intersect({"instrument": {"Bass"}}).force_active() for _ in range(10)]
    # force 1 drums note
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for _ in range(32)]
    # add 40 piano notes
    # e += [ec().intersect({"instrument": {"Piano"}}).force_active() for _ in range(40)]

    # constrain instrument to be only bass and drums
    e += [ec().intersect({"instrument": {"Bass", "Drums"}}).force_active() for i in range(30)]

    # add 50 optional bass and drums
    e += [ec().intersect({"instrument": {"Bass", "Drums", "-"}}) for i in range(30)]
    # pad
    # set tag to pop
    e = [ev.intersect({"tag": {"rock", "-"}}) for ev in e]
    # set tempo to 120
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]

    if len(model.tokenizer.config["midi_types"]) > 0:
        e = [ev.intersect({"midi_type": {"loop", "-"}}) for ev in e]
    return e

e = bass_and_drums([], ec, N_EVENTS)
preview_sm(generate_from_constraints(e, {"temperature": 1.0}))
# %%
# now we'll create a pentatonic chromatic percussion loop

def chromatic_percussion(e, ec, n_events):

    e = [
        ec().intersect({"instrument": {"Chromatic Percussion"}, "duration": {"1/8"}})
        .intersect(ec().pitch_in_scale_constraint("C pentatonic", (60, 108)))
        .force_active() for i in range(50)
    ]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    if len(model.tokenizer.config["midi_types"]) > 0:
        e = [ev.intersect({"midi_type": {"loop", "-"}}) for ev in e]

    return e

e = chromatic_percussion([], ec, N_EVENTS)
preview_sm(generate_from_constraints(e))


#%%

# piano

def piano(e, ec, n_events):

    e = [
        ec().intersect({"instrument": {"Piano"}, "duration": {"1/8"}})
        .intersect(ec().pitch_in_scale_constraint("C major", (60, 108)))
        .force_active() for i in range(50)
    ]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    if len(model.tokenizer.config["midi_types"]) > 0:
        e = [ev.intersect({"midi_type": {"loop", "-"}}) for ev in e]

    return e

e = piano([], ec, N_EVENTS)
preview_sm(generate_from_constraints(e))

# %%
# now piano, just everything piano

def piano(e, ec, n_events):
    
    e = [
        ec().intersect({"instrument": {"Drums", "-"}}) for i in range(N_EVENTS)
    ]
    if len(model.tokenizer.config["midi_types"]) > 0:
        e = [ev.intersect({"midi_type": {"loop", "-"}}) for ev in e]

    return e

e = piano([], ec, N_EVENTS)
sm = generate_from_constraints(e, {"temperature": 1.0})
print(sm.note_num())
preview_sm(sm)


#%%
# now we'll create a drum loop with lots of cowbells
def cowbell_drum_loop(e, ec, n_events):
    e = [
        ec().intersect({"instrument": {"Drums"}, "pitch": {"56"}}).force_active() for i in range(10)
    ]
    # add more drums
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for i in range(40)]
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    if len(model.tokenizer.config["midi_types"]) > 0:
        e = [ev.intersect({"midi_type": {"loop", "-"}}) for ev in e]
    return e

e = cowbell_drum_loop([], ec, N_EVENTS)
preview_sm(generate_from_constraints(e))
# %%
# now we'll create a fast guitar arpeggio loop

def fast_guitar_arpeggio(e, ec, n_events):

    # one guitar note every 16th note

    e = [
        ec().intersect({"instrument": {"Guitar"}, "duration": {"1/16"}, "onset/global_tick": {str(onset_tick)}})
        .intersect(ec().pitch_in_scale_constraint("C major", (48, 84)))
        .force_active() for onset_tick in range(0, 24*16, 6)
    ]

    # set tempo to 120 bpm
    e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    if len(model.tokenizer.config["midi_types"]) > 0:
        e = [ev.intersect({"midi_type": {"loop", "-"}}) for ev in e]
    return e

e = fast_guitar_arpeggio([], ec, N_EVENTS)
preview_sm(generate_from_constraints(e,{"attribute_temperature": {"pitch": 1.0}}))

# we can increase the temperature of pitch 

#%%
# now a drum beat with triplets

def drum_triplets(e, ec, n_events):
    # one drum note every 12th note

    e = [
        ec().intersect({"instrument": {"Drums"}, "onset/global_tick": {str(onset_tick)}}).force_active() for onset_tick in range(0, 24*16, 8)
    ]

    # add 32 more drums
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for i in range(32)]

    # add bass 10 bass notes on any triplet
    triplet_ticks = {str(t) for t in range(0, 24*16, 8)}

    e += [
        ec().intersect({"instrument": {"Bass"}, "onset/global_tick": triplet_ticks}).force_active() for i in range(12)
    ]

    # add guitar on any triplet
    # e += [
    #     ec().intersect({"instrument": {"Guitar"}, "onset/global_tick": triplet_ticks}).force_active() for i in range(12)
    # ]

    # set tempo to 120 bpm
    e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    return e

e = drum_triplets([], ec, N_EVENTS)

sm = generate_from_constraints(e, {"temperature": 1.0})
# preview loop
preview_sm(loop_sm(sm, 4, 2))

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
