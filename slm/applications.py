#%%
import sys
import os
import random
import torch
import symusic

sys.path.append("slm/")
from train import EncoderOnlyModel
from train2 import SuperposedLanguageModel
from util import preview_sm, sm_fix_overlap_notes, loop_sm
from slm.tokenizer import instrument_class_to_selected_program_nr
import util
from paper_checkpoints import checkpoints
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

USE_FP16 = True


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cuda:5"
LLM_DEVICE = "cuda:4"

USE_LOCAL_LLM = False

ROOT_DIR = "./"

MODEL = "harmonic"

OUTPUT_DIR = ROOT_DIR + "artefacts/examples_4"
TMP_DIR = ROOT_DIR + "artefacts/tmp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# model = (
#     EncoderOnlyModel.load_from_checkpoint(
#         ROOT_DIR + checkpoints[MODEL],
#         map_location=device,
#     )
#     .to(device)
#     .eval()
# )


def seq2events(seq):
    # take list of dictionnaries with string items to instead be musical event constraints

    events = []
    for s in seq:
        event = MusicalEventConstraint(
            {key: {value} for key, value in s.items()},
            model.tokenizer,
        )
        events.append(event)
    return events


model = SuperposedLanguageModel.load_from_checkpoint(
    # "./checkpoints/zesty-dawn-376/last.ckpt",
    "../checkpoints/faithful-wave-417/last.ckpt",
    # "./checkpoints/desert-dust-401/last.ckpt",
    map_location=device,
)

print(model.tokenizer.vocab)


def generate(
    mask,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    tokens_per_step=1,
    attribute_temperature=None,
    order=None,
):
    return model.generate(
        mask,
        temperature=temperature,
        tokens_per_step=tokens_per_step,
        top_p=top_p,
        top_k=top_k,
        order=order,
        attribute_temperature=attribute_temperature,
    )[0].argmax(axis=1)


def preview(sm):
    sm = sm.copy()
    sm = sm_fix_overlap_notes(sm)
    preview_sm(loop_sm(sm, 4, 2))


if USE_FP16:
    model = model.convert_to_half()

# create 128 bpm rock loop with drums, bass, guitar with max 280 notes
N_EVENTS = model.tokenizer.config["max_notes"]

blank_event_dict = {
    attr: {
        token.split(":")[-1]
        for token in model.tokenizer.vocab
        if token.startswith(f"{attr}:")
    }
    for attr in model.tokenizer.note_attribute_order
}
ec = lambda: MusicalEventConstraint(blank_event_dict, model.tokenizer)

def sm_to_events(x_sm):
    x = model.tokenizer.encode(x_sm, tag="other")
    tokens = model.tokenizer.indices_to_tokens(x)
    # group by n_attributes
    n_attributes = len(model.tokenizer.note_attribute_order)
    # n_events = model.tokenizer.config["max_notes"]
    events = []
    for i in range(0, len(tokens), n_attributes):
        event = {key: set() for key in model.tokenizer.note_attribute_order}
        for j in range(n_attributes):
            token = tokens[i + j]
            key, value = token.split(":")
            event[key].add(value)
        events.append(event)
    # create event objects
    events = [ec().intersect(event) for event in events]
    return events

#%%


# add some chords
def add_chords(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    tag = "pop"

    instrument = "Piano"

    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # set all tags to jazz
    for i in range(len(e)):
        e[i].a["tag"] = {tag}

    # # remove piano
    # e = [ev for ev in e if ev.a["instrument"].isdisjoint({instrument})]

    # #add a 3 Guitar notes on first beat
    # e += [
    #     ec()
    #     .intersect(
    #         {
    #             "instrument": {instrument},
    #             # constrain to upbeats (meaning onsets on every 12th tick)
    #             "onset/global_tick": {str(i) for i in range(0, 24*4*4, 8)},
    #             # "offset/global_tick": {"96"},
    #             # "duration" : {"1/4", "1/2"},
    #             "duration" : {"1/8"}
    #         }  #   | scale_constraint("C major", (50,100))
    #     )
    #     .force_active()
    #     for i in range(30)
    # ]

    # add 40 drums
    e += [
        ec()
        .intersect(
            {
                "instrument": {"Drums"},
                # "pitch": PERCUSSION_PITCHES,
                # "onset/global_tick": {str(i) for i in range(0, 24 * 4 * 4, 8)},
                "duration" : {"none (Drums)"}
            }
        )
        .force_active()
        for i in range(40)
    ]

    # add 100 optional drums
    e += [
        ec()
        .intersect(
            {
                "instrument": {"Drums", ""},
                # "onset/global_tick": {str(i) for i in range(0, 24 * 4 * 4, 8)},
                "duration" : {"none (Drums)"}
            }
        )
        for i in range(40)
    ]

    # add 4 bass notes with duration 1/4
    e += [
        ec()
        .intersect(
            {
                "instrument": {"Bass"},
                # "onset/global_tick": {str(i) for i in range(0, 24 * 4 * 4, 24)},
                "duration": {"1/4", "1/8"},
            }
        )
        .force_active()
        for i in range(16)
    ]

    # # add 10 optional bass notes
    # e += [
    #     ec()
    #     .intersect(
    #         {
    #             "instrument": {"Bass", ""},
    #             # "onset/global_tick": {str(i) for i in range(0, 24 * 4 * 4, 24)},
    #             # "duration": {"1/4"},
    #         }
    #     )
    #     for i in range(10)
    # ]

    # # add short 20 short guitar notes
    e += [
        ec()
        .intersect(
            {
                "instrument": {"Guitar"},
                "onset/global_tick": {str(i) for i in range(0, 24 * 4 * 4, 12)},
                "duration": {"1/16", "1/8"},
            }
        )
        .intersect(ec().velocity_constraint(40))
        .force_active()
        for i in range(20)
    ]


    # e += [
    #     ec()
    #     .intersect(
    #         {
    #             "instrument": {"Guitar"},
    #             "onset/global_tick": {str(i) for i in range(6, 24 * 4 * 4, 12)},
    #             "duration": {"1/16", "1/8"},
    #         }
    #     ).intersect(
    #         ec().velocity_constraint(120)
    #     )
        
    #     .force_active()
    #     for i in range(20)
    # ]

    # add 10 guitar notes

    # e += [


    # # add 10 notes with duration 1/16
    # e += [
    #     ec()
    #     .intersect(
    #         {
    #             "instrument": {"Bass"},
    #             "duration": {"1/16"},
    #         }
    #     )
    #     .force_active()
    #     for i in range(20)
    # ]

    # add 10 brass
    # e += [
    #     ec()
    #     .intersect(
    #         {
    #             "instrument": {"Brass"},
    #             "duration": {"1/16"},
    #         }
    #     )
    #     .force_active()
    #     for i in range(10)
    # ]


    # 
    

    # add optionalinstrumentnotes
    # e += [ec().intersect({"instrument": {instrument, ""}}) for _ in range(40)]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    # add tag constraint
    e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

    e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

    return e


midi = symusic.Score("../test_midi/C-D-Em-Bm.mid")

TEMPO = 120
# add 4/4 time signature
midi.time_signatures =  [symusic.TimeSignature(0, 4,4)]
midi.tempos = [symusic.Tempo(0, TEMPO)]

print(midi.tempos)

# convert to events
e = sm_to_events(midi)

for ev in e:
    print(ev)


e = add_chords(
    e=e,
    ec=ec,
    n_events=N_EVENTS,
    beat_range=None,
    pitch_range=None,
    drums=False,
    tag=None,
    tempo=TEMPO,
)

TOP_P = 0.95
TEMPERATURE = 1.0
TOP_K = 100
TOKENS_PER_STEP = 1


mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)

x = generate(
    mask,
    top_p=TOP_P,
    temperature=TEMPERATURE,
    top_k=int(TOP_K),
    tokens_per_step=TOKENS_PER_STEP,
    order="random",
    # attribute_temperature={"pitch":0.85},
)
x_sm = model.tokenizer.decode(x)

x_sm = util.sm_fix_overlap_notes(x_sm)

preview(x_sm)

# %%
