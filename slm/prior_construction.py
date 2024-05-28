#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import preview_sm, sm_fix_overlap_notes
import os
import IPython.display as ipd
from paper_checkpoints import checkpoints
import torch

device = "cuda:7"
ROOT_DIR = "../"

MODEL = "slm"

OUTPUT_DIR = ROOT_DIR + "artefacts/examples_4"
TMP_DIR = ROOT_DIR + "artefacts/tmp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + checkpoints[MODEL],
        map_location=device,
    )
    .to(device)
    .eval()
)
#%%

ALL_INSTRUMENTS = {token.split(":")[-1] for token in model.tokenizer.vocab if "instrument:" in token and token != "instrument:-"}
ALL_TEMPOS = {token.split(":")[-1] for token in model.tokenizer.vocab if "tempo:" in token and token != "tempo:-"}
ALL_ONSET_BEATS = {token.split(":")[-1] for token in model.tokenizer.vocab if "onset/beat:" in token and token != "onset/beat:-"}
ALL_ONSET_TICKS = {token.split(":")[-1] for token in model.tokenizer.vocab if "onset/tick:" in token and token != "onset/tick:-"}
ALL_OFFSET_BEATS = {token.split(":")[-1] for token in model.tokenizer.vocab if "offset/beat:" in token and token != "offset/beat:-"}
ALL_OFFSET_TICKS = {token.split(":")[-1] for token in model.tokenizer.vocab if "offset/tick:" in token and token != "offset/tick:-"}
ALL_VELOCITIES = {token.split(":")[-1] for token in model.tokenizer.vocab if "velocity:" in token and token != "velocity:-"}
ALL_TAGS = {token.split(":")[-1] for token in model.tokenizer.vocab if "tag:" in token and token != "tag:-"}
ALL_PITCH = {token.split(":")[-1] for token in model.tokenizer.vocab if "pitch:" in token and token != "pitch:-"}

n_events = model.tokenizer.config["max_notes"]

def create_dead_events():
    events = [
        {
            "instrument": {"-"},
            "pitch": {"-"},
            "onset/beat": {"-"},
            "onset/tick": {"-"},
            "offset/tick": {"-"},
            "offset/beat": {"-"},
            "velocity": {"-"},
            "tag": {"-"},
            "tempo": {"-"},
        } for _ in range(n_events)
    ]
    return events

def is_dead(event):
    for key in event:
        if event[key] != {"-"}:
            return False
    return True

def transform_rejoin(data, filter_fn, map_fn, limit):
    transformed_events = []
    non_transformed_events = []
    idx = 0
    while idx < len(data):
        event = data[idx]
        if len(transformed_events) < limit and filter_fn(event):
            transformed_event = map_fn(event)
            transformed_events.append(transformed_event)
        else:
            non_transformed_events.append(event)
        idx += 1
    return transformed_events + non_transformed_events

def at_least_n_of_x(events, prototype, n):
    # 
    # filter_fn = lambda x: [attribute in x[key] for key, attribute in prototype.items()].all()
    def respects_prototype(event, prototype):
        for key in prototype.keys():
            # check if intersection is not empty
            if not prototype[key] & event[key]:
                return False
        return True
    
    def overwrite_from_prototype(event, prototype):
        for key, attribute in event.items():
            if key in prototype:
                event[key] = prototype[key]
            else:
                attr = event[key]
                # remove "-"
                attr.discard("-")
                event[key] = attribute
        return event
    return transform_rejoin(events, lambda x: respects_prototype(x, prototype), lambda x: overwrite_from_prototype(x, prototype), n)


    


#%%
# create 128 bpm rock loop with drums, bass, guitar with max 280 notes
e = create_dead_events()
filter_fn = lambda x: True
map_fn = lambda x: {
    "instrument": {"Drums", "Bass", "Guitar", "-"},
    "pitch": ALL_PITCH | {"-"},
    "onset/beat": ALL_ONSET_BEATS | {"-"},
    "onset/tick": ALL_ONSET_TICKS | {"-"},
    "offset/beat": ALL_OFFSET_BEATS | {"-"},
    "offset/tick": ALL_OFFSET_TICKS | {"-"},
    "velocity": ALL_VELOCITIES | {"-"},
    "tag": {"rock", "-"},
    "tempo": {"126", "-"},
}

e = transform_rejoin(e, filter_fn, map_fn, 280)
e = at_least_n_of_x(e, {"instrument": {"Drums"}}, 5)
e = at_least_n_of_x(e, {"instrument": {"Bass"}}, 5)
e = at_least_n_of_x(e, {"instrument": {"Guitar"}}, 5)

mask = model.tokenizer.create_mask(e).to(device)

x = model.generate(mask, temperature=0.98)[0].argmax(axis=1)
x_sm = model.tokenizer.decode(x)
x_sm = sm_fix_overlap_notes(x_sm)
preview_sm(x_sm)


#%% replace bassline


# %%
# make ambient loop with 3 instruments
e = create_dead_events()
filter_fn = lambda x: True
map_fn = lambda x: {
    "instrument": {"Piano", "Guitar", "Bass", "-"},
    "pitch": ALL_PITCH | {"-"},
    "onset/beat": ALL_ONSET_BEATS | {"-"},
    "onset/tick": ALL_ONSET_TICKS | {"-"},
    "offset/beat": ALL_OFFSET_BEATS | {"-"},
    "offset/tick": ALL_OFFSET_TICKS | {"-"},
    "velocity": ALL_VELOCITIES | {"-"},
    "tag": {"other", "-"},
    "tempo": {"126", "-"},
}

e = transform_rejoin(e, filter_fn, map_fn, 280)
e = at_least_n_of_x(e, {"instrument": {"Piano"}}, 5)
e = at_least_n_of_x(e, {"instrument": {"Guitar"}}, 5)
e = at_least_n_of_x(e, {"instrument": {"Bass"}}, 5)

mask = model.tokenizer.create_mask(e).to(device)

x = model.generate(mask, temperature=0.97)[0].argmax(axis=1)
x_sm = model.tokenizer.decode(x)
x_sm = sm_fix_overlap_notes(x_sm)
preview_sm(x_sm)
# %%
