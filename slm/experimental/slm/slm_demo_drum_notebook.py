#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from slm.train_old import EncoderOnlyModel
from util import preview_sm, sm_fix_overlap_notes, get_scale, loop_sm
import os
import IPython.display as ipd
from paper_checkpoints import checkpoints
from simplex_diffusion import SimplexDiffusionModel
import torch

device = "cuda:6"
ROOT_DIR = "../"

MODEL = "slm_clean_drums"

OUTPUT_DIR = ROOT_DIR + "artefacts/examples_4"
TMP_DIR = ROOT_DIR + "artefacts/tmp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if MODEL == "slm":
    model = (
        EncoderOnlyModel.load_from_checkpoint(
            ROOT_DIR + checkpoints[MODEL],
            map_location=device,
        )
        .to(device)
        .eval()
    )
    generate = lambda mask: model.generate(mask, temperature=1.0, top_p=0.95)[0].argmax(
        axis=1
    )
elif MODEL == "slm_harmonic":
    model = (
        EncoderOnlyModel.load_from_checkpoint(
            # ROOT_DIR + "checkpoints/ruby-surf-331/epoch=1119-step=17920-val/loss_epoch=0.03922.ckpt",
            ROOT_DIR
            + "checkpoints/fanciful-star-356/epoch=1199-step=81600-val/loss_epoch=0.10647.ckpt",
            map_location=device,
        )
        .to(device)
        .eval()
    )
    generate = lambda mask: model.generate(
        mask,
        temperature=1.0,
        top_p=0.98,
    )[0].argmax(axis=1)
elif MODEL == "slm_clean_drums":
    model = (
        EncoderOnlyModel.load_from_checkpoint(
            # ROOT_DIR
            # + "checkpoints/sparkling-violet-330/epoch=159-step=7200-val/loss_epoch=0.04531.ckpt",
            # ROOT_DIR
            # + "checkpoints/fresh-grass-346/last.ckpt",
             ROOT_DIR + "checkpoints/comfy-morning-351/epoch=2199-step=288200-val/loss_epoch=0.11651.ckpt",
            # ROOT_DIR + "checkpoints/vocal-energy-350/epoch=899-step=117900-val/loss_epoch=0.10755.ckpt",
            # ROOT_DIR +"checkpoints/magic-star-347/last.ckpt",
            # ROOT_DIR + "checkpoints/bumbling-wave-348/last.ckpt",
            # ROOT_DIR + "checkpoints/generous-donkey-335/last.ckpt",
            map_location="cpu",
        )
        .to(device)
        .eval()
    )
    def generate(mask,temperature=1.0,top_p=1.0,attribute_temperature=None): 
        return model.generate(
        mask,
        temperature=temperature,
        top_p=top_p,
        # attribute_temperature={"velocity": 1.5,"onset/tick":0.5},
        )[0].argmax(axis=1)

else:
    model = SimplexDiffusionModel.load_from_checkpoint(
        # "../checkpoints/dark-sky-67/last.ckpt", map_location=device
        # f"../checkpoints/valiant-sea-3/last.ckpt",
        ROOT_DIR + "checkpoints/worldly-plant-13/last.ckpt",
        map_location="cpu"
    ).to(device)
    # model = SimplexDiffusionModel.load_from_checkpoint(
    #     "../checkpoints/flowing-paper-64/last.ckpt", map_location=device
    # )

    def generate(mask, temperature=1.0, top_p=1.0, steps=100):
        return model.sample2(
            mask,
            enforce_prior=True,
            nb_steps=steps,
            top_p=top_p,
            batch_size=1,
            prior_strength=1,

    )[0]


def preview(sm):
    sm = sm.copy()
    sm = sm_fix_overlap_notes(sm)
    preview_sm(loop_sm(sm, 4, 4))

# %%
# create 128 bpm rock loop with drums, bass, guitar with max 280 notes and minimum 2 drum notes and maximum 40 drum notes

ALL_INSTRUMENTS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "instrument:" in token and token != "instrument:-"
}

ALL_TEMPOS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "tempo:" in token and token != "tempo:-"
}
ALL_ONSET_BEATS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "onset/beat:" in token and token != "onset/beat:-"
}
ALL_ONSET_TICKS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "onset/tick:" in token and token != "onset/tick:-"
}
ALL_OFFSET_BEATS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "offset/beat:" in token and token != "offset/beat:-"
}
ALL_OFFSET_TICKS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "offset/tick:" in token and token != "offset/tick:-"
}
ALL_VELOCITIES = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "velocity:" in token and token != "velocity:-"
}
ALL_TAGS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "tag:" in token and token != "tag:-"
}
ALL_PITCH = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "pitch:" in token and token != "pitch:-"
}
# all pitches that contain "(Drums)"
DRUM_PITCHES = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "pitch:" in token and "(Drums)" in token
}

HIHAT_PITCHES = {
    f"{pitch} (Drums)" for pitch in ["42", "44", "46"]
}

TOM_PITCHES = {
    f"{pitch} (Drums)" for pitch in ["48", "50", "45", "47"]
}

CRASH_PITCHES = {
    f"{pitch} (Drums)" for pitch in ["49", "57"]
}

PERCUSSION_PITCHES = {
    f"{pitch} (Drums)" for pitch in ["60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71","72", "73", "74", "75", "76", "77", "78", "79", "80", "81"]
}

# create 128 bpm rock loop with drums, bass, guitar with max 280 notes
N_EVENTS = model.tokenizer.config["max_notes"]


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
    events = [EventConstraint().intersect(event) for event in events]
    return events


class EventConstraint:
    def __init__(self):
        self.blank_event = {
            "instrument": ALL_INSTRUMENTS | {"-"},
            "pitch": ALL_PITCH | {"-"},
            "onset/beat": ALL_ONSET_BEATS | {"-"},
            "onset/tick": ALL_ONSET_TICKS | {"-"},
            "offset/beat": ALL_OFFSET_BEATS | {"-"},
            "offset/tick": ALL_OFFSET_TICKS | {"-"},
            "velocity": ALL_VELOCITIES | {"-"},
            "tag": ALL_TAGS | {"-"},
            "tempo": ALL_TEMPOS | {"-"},
        }

        self.a = self.blank_event.copy()

        self.all_active = {
            "instrument": ALL_INSTRUMENTS,
            "pitch": ALL_PITCH,
            "onset/beat": ALL_ONSET_BEATS,
            "onset/tick": ALL_ONSET_TICKS,
            "offset/beat": ALL_OFFSET_BEATS,
            "offset/tick": ALL_OFFSET_TICKS,
            "velocity": ALL_VELOCITIES,
            "tag": ALL_TAGS,
            "tempo": ALL_TEMPOS,
        }

        self.not_active = {
            "instrument": {"-"},
            "pitch": {"-"},
            "onset/beat": {"-"},
            "onset/tick": {"-"},
            "offset/beat": {"-"},
            "offset/tick": {"-"},
            "velocity": {"-"},
            "tag": {"-"},
            "tempo": {"-"},
        }

    def intersect(self, constraint):
        for key in constraint:
            self.a[key] = self.a[key] & constraint[key]
        for key in self.a:
            assert (
                len(self.a[key]) > 0
            ), f"Empty set for key {key}, constraint {constraint}, attributes {self.a}"
        return self

    def union(self, constraint):
        for key in constraint:
            self.a[key] = self.a[key] | constraint[key]
        return self

    def is_inactive(self):
        for key in self.a:
            if self.a[key] != {"-"}:
                return False
        return True

    def is_active(self):
        for key in self.a:
            if self.a[key] == {"-"}:
                return False
        return True

    def force_active(self):
        self.intersect(self.all_active)
        return self

    def force_inactive(self):
        self.intersect(self.not_active)
        return self

    def to_dict(self):
        return self.a

def scale_constraint(scale, pitch_range):
    scale_constraint = {
        "pitch": {str(note) for note in get_scale(scale, pitch_range)} | {"-"}
    }
    return scale_constraint

def tempo_constraint(tempo):
    # find tempo that is closest to the given tempo
    tempos = list(str(t) for t in ALL_TEMPOS)
    tempo = min(tempos, key=lambda x: abs(int(x) - tempo))
    return {"tempo": {tempo, "-"}}

def velocity_constraint(velocity):
    velocities = list(str(v) for v in ALL_VELOCITIES)
    velocity = min(velocities, key=lambda x: abs(int(x) - velocity))
    return {"velocity": {velocity, "-"}}


#%%
# create simple drumbeat with 40 notes
def simple_beat():
    e = [EventConstraint().force_active() for _ in range(60)]
    # pad 
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # tempo to 96 and tag is funk
    e = [ev.intersect(tempo_constraint(96)|{"tag": {"funk", "-"}}) for ev in e]
    return e

e = simple_beat()
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = generate(mask, top_p=0.99,temperature=1.0)
x_sm = model.tokenizer.decode(x)
print(x_sm.note_num())
preview(x_sm)

#%%
# more dynamic hihats

e = sm_to_events(x_sm)
def with_dynamic_hihats(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]
    # remove hihats
    e = [ev for ev in e if ev.a["pitch"].isdisjoint(HIHAT_PITCHES)]

    e += [EventConstraint().intersect({"pitch": HIHAT_PITCHES} | velocity_constraint(30)).force_active() for _ in range(3)]

    e += [EventConstraint().intersect({"pitch": HIHAT_PITCHES} | velocity_constraint(60)).force_active() for _ in range(3)]

    # add up to 10 more
    e += [EventConstraint().intersect({"pitch": HIHAT_PITCHES | {"-"}}) for _ in range(30)]

    # pad
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    return e

e = with_dynamic_hihats(e)
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = generate(mask, top_p=0.96)
x_sm = model.tokenizer.decode(x)
print(x_sm.note_num())
preview(x_sm)

#%%

# add snare ghost notes
e = sm_to_events(x_sm)

def add_snare_ghost_notes(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    e += [EventConstraint().intersect({"pitch": {"38 (Drums)","-"}}|velocity_constraint(50)).force_active() for _ in range(5)]

    # pad
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    return e

e = add_snare_ghost_notes(e)
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = generate(mask, top_p=0.95)
x_sm = model.tokenizer.decode(x)
print(x_sm.note_num())
preview(x_sm)

#%%

# add percussion
e = sm_to_events(x_sm)

def add_percussion(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    e += [
        EventConstraint()
        .intersect({"pitch": PERCUSSION_PITCHES | {"-"}})
        .force_active()
        for _ in range(5)
    ]

    e += [
        EventConstraint().intersect({"pitch": PERCUSSION_PITCHES | {"-"}})
        for _ in range(20)
    ]
    # pad
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    return e


e = add_percussion(e)
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = generate(mask, top_p=0.95)
x_sm = model.tokenizer.decode(x)
print(x_sm.note_num())
preview(x_sm)

#%%

# add fill

e = sm_to_events(x_sm)

def with_tom_fill(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    e = [ev for ev in e if ev.a["onset/beat"].isdisjoint({"12","13","14","15"})]

    # add at least one tom of each pitch
    e += [EventConstraint().intersect({"pitch": TOM_PITCHES, "onset/beat": {"12","13","14","15","_"}} ).force_active() for e in range(3)]

    # add up to 10 more
    e += [EventConstraint().intersect({"onset/beat": {"12","13","14","15","_"}}) for _ in range(20)]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    return e

e = with_tom_fill(e)
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = generate(mask, top_p=0.97)
x_sm = model.tokenizer.decode(x)
print(x_sm.note_num())
preview(x_sm)

#%%

# reinstrument

e = sm_to_events(x_sm)

def reinstrument(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    e = [ev.union({"pitch":PERCUSSION_PITCHES}).intersect({"pitch":PERCUSSION_PITCHES}).force_active() for ev in e]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    return e

e = reinstrument(e)
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = generate(mask, temperature=1.4)
x_sm = model.tokenizer.decode(x)
print(x_sm.note_num())
preview(x_sm)

#%%
# create 4 on the floor beat

def four_on_the_floor_beat():
    e = []
    # add kick on every beat
    for onset_beat in ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]:
        e += [
            EventConstraint()
            .intersect(
                {"pitch": {"36 (Drums)"}, "onset/beat": {onset_beat}, "onset/tick": {"0"}}
            )
            .force_active()
        ]
    # snares on 2 and 4
    for onset_beat in ["1","3","5","7","9","11","13","15"]:
        e += [
            EventConstraint()
            .intersect(
                {"pitch": {"38 (Drums)"}, "onset/beat": {onset_beat}, "onset/tick": {"0"}}
            )
            .force_active()
        ]
    # add 40 hihats
    e += [
        EventConstraint().intersect(
            {"pitch": HIHAT_PITCHES | {"-"}}
        ) for _ in range(20)
    ]
    # add percussion
    e += [
        EventConstraint().intersect(
            {"pitch": PERCUSSION_PITCHES | {"-"}}
        ) for _ in range(20)
    ]
    e += [EventConstraint() for _ in range(N_EVENTS - len(e))]
    # set tempo to 110
    e = [ev.intersect(tempo_constraint(130)).
         intersect({"instrument":{"Drums","-"}})
         for ev in e]
    return e

e = four_on_the_floor_beat()
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = generate(mask, temperature=1.0)
x_sm = model.tokenizer.decode(x)
print(x_sm.note_num())
preview(x_sm)


#%%

# create breakbeat
def breakbeat():
    e = []
    # add 10 kicks
    e += [
        EventConstraint()
        .intersect(
            {"pitch": {"36 (Drums)"}}
        )
        .force_active()
        for _ in range(10)
    ]
    # add 10 optional kicks
    e += [
        EventConstraint()
        .intersect(
            {"pitch": {"36 (Drums)", "-"}}
        )
        for _ in range(10)
    ]
    # add 20 rides
    e += [
        EventConstraint().intersect({"pitch": {"51 (Drums)"}}).force_active()
        for _ in range(20)
    ]
    # 20 optional rides
    e += [
        EventConstraint().intersect({"pitch": {"51 (Drums)", "-"}})
        for _ in range(20)
    ]
    # add 10 snare
    e += [
        EventConstraint()
        .intersect(
            {"pitch": {"38 (Drums)"}}
        )
        .force_active()
        for _ in range(10)
    ]
    # add 10 optional snares
    e += [
        EventConstraint()
        .intersect(
            {"pitch": {"38 (Drums)", "-"}}
        )
        for _ in range(10)
    ]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # set to 160
    e = [ev.intersect(tempo_constraint(160)).
         intersect({"instrument":{"Drums","-"}})
         for ev in e]
    return e

e = breakbeat()
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = generate(mask, temperature=1.0)
x_sm = model.tokenizer.decode(x)
print(x_sm.note_num())
preview(x_sm)


#%%
# create

#%%
# add dynamics

e = sm_to_events(x_sm)

def new_dynamics(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]
    e = [ev.union({"velocity": ALL_VELOCITIES}).force_active() for ev in e]
    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e

e = new_dynamics(e)
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = generate(mask, temperature=1.0)
x_sm = model.tokenizer.decode(x)
print(x_sm.note_num())
preview(x_sm)

#%%
e = sm_to_events(x_sm)

def new_timing(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]
    e = [ev.union({"onset/tick": ALL_ONSET_TICKS}).force_active() for ev in e]
    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e

e = new_timing(e)
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = generate(mask, temperature=1.0)
x_sm = model.tokenizer.decode(x)
print(x_sm.note_num())
preview(x_sm)

#%%
# make sparser


# make denser

# create syncopated beat

# reinstrumentation

# create a drum solo

# add chords on snares

# create 4 on the floor beat



# create reggae beat

# create breakbeat

# add bassline that matches kick

# full instrumentation

# create jazz loop

# add pipe melody

# add harmony

# constrain to a scale, imply root

# replace instruments

# replace pitches

# replace onsets/offsets

# jazz up chords

# add chords

# add call and response structure

# infill pitch time






# %%
