import sys

sys.path.append("slm/")
import socket
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from train import EncoderOnlyModel
from util import preview_sm, sm_fix_overlap_notes, get_scale, loop_sm
import os
import IPython.display as ipd
from paper_checkpoints import checkpoints
from simplex_diffusion import SimplexDiffusionModel
import torch
import tempfile
import util

remdevice = "cuda:6"

USE_FP16 = True


def get_external_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    external_ip = s.getsockname()[0]
    s.close()
    return external_ip


print("External IP Address:", get_external_ip())

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
# allow all origins
CORS(app)

import datetime
import random

# from lazy_drums.playback_engine import PlaybackEngine
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi as pm
import pytorch_lightning as pl
import seaborn as sns
import torch



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cuda:6"
ROOT_DIR = "./"

MODEL = "slm"

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

    def generate(
        mask, temperature=1.0, top_p=1.0,top_k=0, tokens_per_step=1, attribute_temperature=None, order=None
    ):
        return model.generate(
            mask,
            temperature=temperature,
            tokens_per_step=tokens_per_step,
            top_p=top_p,
            top_k=top_k,
            order=order,
            # attribute_temperature={"velocity": 1.5,"onset/tick":0.5},
        )[0].argmax(axis=1)
elif MODEL == "slm_harmonic":
    # TODO: Figure out which checkpoint sounds best
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

    def generate(
        mask, temperature=1.0, top_p=1.0, attribute_temperature=None, order=None
    ):
        return model.generate(
            mask,
            temperature=temperature,
            top_p=top_p,
            order=order,
            # attribute_temperature={"velocity": 1.5,"onset/tick":0.5},
        )[0].argmax(axis=1)
elif MODEL == "slm_clean_drums":
    model = (
        EncoderOnlyModel.load_from_checkpoint(
            # ROOT_DIR
            # + "checkpoints/sparkling-violet-330/epoch=159-step=7200-val/loss_epoch=0.04531.ckpt",
            # ROOT_DIR
            # + "checkpoints/fresh-grass-346/last.ckpt",
            # ROOT_DIR
            # + "checkpoints/comfy-morning-351/epoch=2199-step=288200-val/loss_epoch=0.11651.ckpt",
            ROOT_DIR + "checkpoints/vocal-energy-350/epoch=899-step=117900-val/loss_epoch=0.10755.ckpt",
            # ROOT_DIR +"checkpoints/magic-star-347/last.ckpt",
            # ROOT_DIR + "checkpoints/bumbling-wave-348/last.ckpt",
            # ROOT_DIR + "checkpoints/generous-donkey-335/last.ckpt",
            map_location="cpu",
        )
        .to(device)
        .eval()
    )

    def generate(mask, temperature=1.0, top_p=1.0, top_k=1.0,attribute_temperature=None, order=None):
        return model.generate(
            mask,
            temperature=temperature,
            top_p=top_p,
            order=order,
            # attribute_temperature={"velocity": 1.5,"onset/tick":0.5},
        )[0].argmax(axis=1)

else:
    model = SimplexDiffusionModel.load_from_checkpoint(
        # "../checkpoints/dark-sky-67/last.ckpt", map_location=device
        # f"../checkpoints/valiant-sea-3/last.ckpt",
        ROOT_DIR + "checkpoints/worldly-plant-13/last.ckpt",
        map_location="cpu",
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


if USE_FP16:
    model = model.convert_to_half()

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

HIHAT_PITCHES = {f"{pitch} (Drums)" for pitch in ["42", "44", "46"]}

TOM_PITCHES = {f"{pitch} (Drums)" for pitch in ["48", "50", "45", "47"]}

CRASH_PITCHES = {f"{pitch} (Drums)" for pitch in ["49", "57"]}

PERCUSSION_PITCHES = {
    f"{pitch} (Drums)"
    for pitch in [
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
    ]
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
    def __init__(self, dict=None):
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

        if dict is not None:
            self.a = dict

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

def quantize_velocity(velocity):
    velocities = list(str(v) for v in ALL_VELOCITIES)
    velocity = min(velocities, key=lambda x: abs(int(x) - int(velocity)))
    return velocity

def quantize_tempo(tempo):
    tempos = list(str(t) for t in ALL_TEMPOS)
    tempo = min(tempos, key=lambda x: abs(int(x) - int(tempo)))
    return tempo

@app.route("/", methods=["GET"])
def index():
    return "Hello, World!"

def simple_beat():
    e = [EventConstraint().force_active() for _ in range(80)]
    # pad
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # tempo to 96 and tag is funk
    e = [ev.intersect(tempo_constraint(148) | {"tag": {"funk", "-"}}) for ev in e]
    return e


def four_on_the_floor_beat():
    e = []
    # add kick on every beat
    for onset_beat in [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"36 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # snares on 2 and 4
    for onset_beat in ["1", "3", "5", "7", "9", "11", "13", "15"]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"38 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # add 40 hihats
    e += [
        EventConstraint().intersect({"pitch": HIHAT_PITCHES | {"-"}}) for _ in range(80)
    ]
    # add percussion
    e += [
        EventConstraint().intersect({"pitch": PERCUSSION_PITCHES | {"-"}})
        for _ in range(20)
    ]
    e += [EventConstraint() for _ in range(N_EVENTS - len(e))]
    # set tempo to 110
    e = [
        ev.intersect(tempo_constraint(130)).intersect({"instrument": {"Drums", "-"}})
        for ev in e
    ]
    return e

# create breakbeat
def breakbeat():
    e = []
    # add 10 kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)"}}).force_active()
        for _ in range(10)
    ]
    # add 10 optional kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)", "-"}}) for _ in range(10)
    ]
    # add 3 toms
    e += [
        EventConstraint().intersect({"pitch": TOM_PITCHES}).force_active()
        for _ in range(10)
    ]

    # add 20 rides
    e += [
        EventConstraint().intersect({"pitch": {"51 (Drums)"}}).force_active()
        for _ in range(20)
    ]
    # 20 optional rides
    e += [
        EventConstraint().intersect({"pitch": {"51 (Drums)", "-"}}) for _ in range(20)
    ]
    # add 10 snare
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)"}}).force_active()
        for _ in range(10)
    ]
    # add 10 optional snares
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)", "-"}}) for _ in range(10)
    ]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # set to 160
    e = [
        ev.intersect(tempo_constraint(150)).intersect({"instrument": {"Drums", "-"}})
        for ev in e
    ]
    return e

def funk_beat():

    e = []

    # add 10 kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)"}}).force_active()
        for _ in range(20)
    ]

    # add 4 snares
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)"}}).force_active()
        for _ in range(4)
    ]

    # add 10 hihats
    e += [
        EventConstraint().intersect({"pitch": {"42 (Drums)"}}).force_active()
        for _ in range(40)
    ]

    # add 4 open
    e += [
        EventConstraint().intersect({"pitch": {"46 (Drums)"}}).force_active()
        for _ in range(4)
    ]

    # add 10 ghost snare
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)"}}).intersect(velocity_constraint(40)).force_active()
        for _ in range(10)
    ]

    # add up to 20 optional drum notes
    e += [EventConstraint().intersect({"instrument": {"Drums"}}) for _ in range(20)]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    # set to 96
    e = [ev.intersect(tempo_constraint(96)) for ev in e]

    # set tag to funk
    e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]

    return e




def synth_beat():
    e = []
    # add 10 bass
    e += [
        EventConstraint().intersect({"instrument": {"Bass"}}).force_active()
        for _ in range(10)
    ]

    # add 40 synth lead, 4 per beat
    for beat in range(16):
        for tick in range(4):
            e += [
                EventConstraint().intersect({"instrument": {"Synth Lead"}}).intersect(
                    {"onset/beat": {str(beat)} , "onset/tick": {str(model.tokenizer.config["ticks_per_beat"] // 4 * tick)}}
                ).force_active()
            ]
      
    
    # add 2 forced synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Synth Lead"}}).force_active()
        for _ in range(2)
    ]

    # add 10 piano
    e += [
        EventConstraint().intersect({"instrument": {"Piano"}}).force_active()
        for _ in range(10)
    ]

    # add 20 drums
    e += [
        EventConstraint().intersect({"instrument": {"Drums"}}).force_active()
        for _ in range(50)
    ]

    # add 50 optional bass or synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Bass", "Drums", "Piano", "-"}})
        for _ in range(100)
    ]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # set to 125
    e = [
        ev.intersect(tempo_constraint(125))
        for ev in e
    ]
    # set tag to pop
    e = [ev.intersect({"tag": {"other", "-"}}) for ev in e]

    return e

# create breakbeat
def prog_beat():
    e = []
    # add 10 kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)"}}).force_active()
        for _ in range(10)
    ]
    # add 10 optional kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)", "-"}}) for _ in range(10)
    ]
    # add 3 toms
    # e += [
    #     EventConstraint().intersect({"pitch": TOM_PITCHES}).force_active()
    #     for _ in range(10)
    # ]

    # add 20 rides
    e += [
        EventConstraint().intersect({"pitch": {"42 (Drums)"}}).force_active()
        for _ in range(20)
    ]
    # 20 optional rides
    e += [
        EventConstraint().intersect({"pitch": {"42 (Drums)", "-"}}) for _ in range(20)
    ]
    # add 10 snare
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)"}}).force_active()
        for _ in range(10)
    ]
    # add 10 optional snares
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)", "-"}}) for _ in range(10)
    ]

    # add 20 bass notes
    e += [
        EventConstraint().intersect({"instrument": {"Bass"},
                                     "pitch": {str(p) for p in range(36, 48)}}).force_active()
        for _ in range(30)
    ]
    # add 20 piano notes
    e += [
        EventConstraint().intersect({"instrument": {"Piano"}}).force_active()
        for _ in range(20)
    ]

    # add 20 guitar notes
    e += [
        EventConstraint().intersect({"instrument": {"Guitar"}}).force_active()
        for _ in range(20)
    ]

    e += [
        EventConstraint()
        .intersect(tempo_constraint(160))
        .intersect({"instrument": {"Bass", "Drums", "Piano", "Guitar", "-"}})
        for _ in range(N_EVENTS - len(e))
    ]
    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # set to 160
    # add 50 optional notes
    
    e = [
        ev.intersect(tempo_constraint(160)).intersect({"instrument": {"Drums","Bass","Piano","Guitar", "-"}})
        for ev in e
    ]

    # constrain to pentatonic scale
    # e = [ev.intersect(
    #     {"pitch":scale_constraint("C pentatonic", (20, 100))["pitch"] | {"-"} | DRUM_PITCHES}
    #     ) for ev in e]
    return e


def four_on_the_floor_beat():
    e = []
    # add kick on every beat
    for onset_beat in [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"36 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # snares on 2 and 4
    for onset_beat in ["1", "3", "5", "7", "9", "11", "13", "15"]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"38 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # add 40 hihats
    e += [
        EventConstraint().intersect({"pitch": HIHAT_PITCHES | {"-"}}) for _ in range(20)
    ]
    # add percussion
    e += [
        EventConstraint().intersect({"pitch": PERCUSSION_PITCHES | {"-"}})
        for _ in range(20)
    ]
    e += [EventConstraint() for _ in range(N_EVENTS - len(e))]
    # set tempo to 110
    e = [
        ev.intersect(tempo_constraint(130)).intersect({"instrument": {"Drums", "-"}})
        for ev in e
    ]
    return e


def disco_beat():
    e = []
    # add kick on every beat
    for onset_beat in [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"36 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # snares on 2 and 4
    for onset_beat in ["1", "3", "5", "7", "9", "11", "13", "15"]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"38 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # add 40 hihats
    e += [
        EventConstraint().intersect({"pitch": HIHAT_PITCHES | {"-"}}).force_active()
        for _ in range(20)
    ]
    # add percussion
    e += [
        EventConstraint().intersect({"pitch": PERCUSSION_PITCHES | {"-"}})
        for _ in range(20)
    ]

    e += [
        EventConstraint().intersect({"instrument": {"Bass"}}).force_active()
        for _ in range(16)
    ]
    # add 10 piano notes
    e += [
        EventConstraint().intersect({"instrument": {"Piano"}}).force_active()
        for _ in range(20)
    ]
    # add 10 guitar notes
    e += [
        EventConstraint().intersect({"instrument": {"Guitar"}}).force_active()
        for _ in range(10)
    ]
    # add 10 synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Synth Lead"}}).force_active()
        for _ in range(10)
    ]
    # add 50 blank notes
    e += [EventConstraint().force_inactive() for _ in range(50)]
    e += [
        EventConstraint().intersect(
            {"instrument": {"Bass", "Drums", "Piano", "Guitar", "Synth Lead", "-"}}
        )
        for _ in range(N_EVENTS - len(e))
    ]
    # set tempo to 110
    # add 30 bass notes

    e = [
        ev.intersect(tempo_constraint(130)).intersect(
            {"instrument": {"Bass", "Drums", "Piano", "Guitar", "Synth Lead", "-"}}
        )
        for ev in e
    ]
    # add pop tag
    e = [ev.intersect({"tag": {"pop", "-"}}) for ev in e]
    return e

def metal_beat():
    e = []

    # add 10 kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)"}}).force_active()
        for _ in range(12)
    ]

    # add 4 snares
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)"}}).force_active()
        for _ in range(4)
    ]

    # add 10 hihats
    e += [
        EventConstraint().intersect({"pitch": HIHAT_PITCHES}).force_active()
        for _ in range(10)
    ]

    # add up to 20 optional drum notes
    e += [EventConstraint().intersect({"instrument": {"Drums"}}) for _ in range(10)]

    # add 30 guitar notes
    e += [
        EventConstraint()
        .intersect({"instrument": {"Guitar"}})
        # .intersect(scale_constraint("E pentatonic", (30, 100)))
        .force_active()
        for _ in range(50)
    ]

    # add 20 bass notes
    e += [
        EventConstraint()
        .intersect({"instrument": {"Bass"}})
        # .intersect(scale_constraint("E pentatonic", (30, 100)))
        .force_active()
        for _ in range(20)
    ]

    # add 30 optional guitar or bass notes
    e += [
        EventConstraint().intersect({"instrument": {"Guitar", "Bass", "-"}})
        # .intersect(scale_constraint("E pentatonic", (30, 100)))
        for _ in range(30)
    ]

    # add 40 optional notes, guitar drums or bass
    e += [
        EventConstraint().intersect({"instrument": {"Guitar", "Drums", "Bass", "-"}})
        for _ in range(80)
    ]


    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # set tempo to 160
    e = [ev.intersect(tempo_constraint(150)) for ev in e]
    # set tag to metal
    # e = [ev.intersect({"tag": {"metal", "-"}}) for ev in e]

    return e

def fun_beat():

    e = []
    # add 30 piano
    e += [
        EventConstraint().intersect({"instrument": {"Chromatic Percussion"}}).force_active()
        for _ in range(20)
    ]
    # add 10 low velocity Chromatic Percussion
    e += [
        EventConstraint()
        .intersect({"instrument": {"Chromatic Percussion"}})
        .intersect(velocity_constraint(100))
        .intersect({"pitch": {str(s) for s in range(50, 100)}})
        .force_active()
        for _ in range(10)
    ]
    # add 10 high velocity Chromatic Percussion
    e += [
        EventConstraint().intersect({"instrument": {"Chromatic Percussion"}}).intersect(velocity_constraint(40)).force_active()
        for _ in range(10)
    ]

    # add 30 guitar
    # e += [
    #     EventConstraint().intersect({"instrument": {"Guitar"}}).force_active()
    #     for _ in range(30)
    # ]

    # add 5 bass
    # e += [
    #     EventConstraint().intersect({"instrument": {"Bass"}}).force_active()
    #     for _ in range(5)
    # ]

    # add 10 synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Synth Pad"}}).force_active()
        for _ in range(16)
    ]

    # add 50 optional
    e += [
        EventConstraint().intersect({"instrument": {"Bass", "-"}}).intersect(scale_constraint("C pentatonic", (30, 50)))
        for _ in range(20)
    ]
    # add one bass note on first beat
    e += [
        EventConstraint().intersect({"instrument": {"Bass"}}).intersect(scale_constraint("C pentatonic", (30, 50))).force_active() for _ in range(10)
    ]


    # add 40 drums
    

    # constrain to major pitch set
    e = [ev.intersect(scale_constraint("C major", (20, 100))) for ev in e]
    # pad with empty notes
    e += [
        EventConstraint()
        .intersect({"instrument": {"Drums"}, "pitch": PERCUSSION_PITCHES})
        .force_active()
        for _ in range(40)
    ]
    # add 30 drum notes
    e += [
        EventConstraint().intersect({"instrument": {"Drums"}, "pitch": DRUM_PITCHES- PERCUSSION_PITCHES
                                     }).force_active()
        for _ in range(30)
    ]

    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    # set tag to pop
    e = [ev.intersect({"tag": {"other", "-"}}).intersect(tempo_constraint(120)) for ev in e]


    return e



def simple_beat():

    e = []
    # add 30 piano
    e += [
        EventConstraint().intersect({"instrument": {"Piano"}}).force_active()
        for _ in range(20)
    ]
    # add 10 low velocity piano
    e += [
        EventConstraint()
        .intersect({"instrument": {"Piano"}})
        .intersect(velocity_constraint(100))
        .intersect({"pitch": {str(s) for s in range(50, 100)}})
        .force_active()
        for _ in range(10)
    ]
    # add 10 high velocity piano
    e += [
        EventConstraint().intersect({"instrument": {"Piano"}}).intersect(velocity_constraint(40)).force_active()
        for _ in range(10)
    ]

    # add 30 guitar
    # e += [
    #     EventConstraint().intersect({"instrument": {"Guitar"}}).force_active()
    #     for _ in range(30)
    # ]

    # add 5 bass
    # e += [
    #     EventConstraint().intersect({"instrument": {"Bass"}}).force_active()
    #     for _ in range(5)
    # ]

    # add 10 synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Synth Lead"}}).force_active()
        for _ in range(10)
    ]

    # add 50 optional
    e += [
        EventConstraint().intersect({"instrument": {"Piano", "-"}})
        for _ in range(20)
    ]

    # constrain to major pitch set
    e = [ev.intersect(scale_constraint("C major", (20, 100))) for ev in e]
    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    # set tag to pop
    e = [ev.intersect({"tag": {"pop", "-"}}).intersect(tempo_constraint(90)) for ev in e]

    return e


def camel_to_snake(name):
    import re

    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

def json_camel_to_snake(data):
    if isinstance(data, dict):
        return {camel_to_snake(k): json_camel_to_snake(v) for k, v in data.items()}
    if isinstance(data, list):
        return [json_camel_to_snake(v) for v in data]
    return data

def make_sparse(e):
    n_events = model.tokenizer.config["max_notes"]
    # remove empty events
    e = [event for event in e if not event.is_inactive()]
    # make every note optional
    e = [
        event.union(
            {
                "instrument": {"-"},
                "pitch": {"-"},
                "onset/beat": {"-"},
                "onset/tick": {"-"},
                "tempo": {"-"},
                "tag": {"-"},
                "velocity": {"-"},
                "offset/beat": {"-"},
                "offset/tick": {"-"},
            }
        )
        for event in e
    ]

    # shuffle
    # force half of the notes to be active
    # for i in range(n_events):
    #     e[i] = e[i].force_active()
    # pad with empty events
    e += [EventConstraint().force_inactive() for _ in range(n_events - len(e))]
    random.shuffle(e)

    return e

def sm_to_looprep(sm):
    # make two sequences. one for drums and one for other instruments
    drum_sequence = []
    harm_sequence = []
    for track in sm.tracks:
        for note in track.notes:
            if track.name == "Drums":
                drum_sequence.append({
                    "pitch": note.pitch,
                    "onset": note.start,
                    "velocity": note.velocity,
                    "duration": sm.tpq//4,
                    "instrument": track.name
                })
            else:
                harm_sequence.append(
                    {
                        "pitch": note.pitch,
                        "onset": note.start,
                        "velocity": note.velocity,
                        "duration": note.duration,
                        "instrument": track.name,
                    }
                )
    return {
        "time_signature": "4/4",
        "tempo": sm.tempos[0].qpm,
        "n_bars":4,
        "drum_seq": drum_sequence,
        "harm_seq": harm_sequence,
        "ppq": sm.tpq
    }

def seq2events(sequence, tempo):
    events = []
    for note_event in sequence:

        event = EventConstraint(
            {
                "pitch": {str(note_event["pitch"])+" (Drums)" if note_event["instrument"] == "Drums" else str(note_event["pitch"])},
                "onset/beat": {str(note_event["onset"] // model.tokenizer.config["ticks_per_beat"])},
                "onset/tick": {str(note_event["onset"] % model.tokenizer.config["ticks_per_beat"])},
                "velocity": {str(quantize_velocity(int(note_event["velocity"])))},
                "instrument": {note_event["instrument"]},
                "tempo": {str(quantize_tempo(tempo))},
                "tag": {"other"},
                "offset/beat": {str((note_event["onset"] + note_event["duration"]) // model.tokenizer.config["ticks_per_beat"]) if note_event["instrument"] != "Drums" else "none (Drums)"},
                "offset/tick": {str((note_event["onset"] + note_event["duration"]) % model.tokenizer.config["ticks_per_beat"]) if note_event["instrument"] != "Drums" else "none (Drums)"}
            }
        )
        events += [event]
    return events

# with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#         x_sm.dump_midi(tmp_file.name)
#         tmp_file.flush()
#         with open(tmp_file.name, "rb") as f:
#             midi_bytes = f.read()
#     try:
#         os.remove(tmp_file.name)
#     except OSError:
#         pass


def infill(e, beat_range, pitch_range, drums, tag="other", tempo=120):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(quantize_tempo(tempo))}

    beats = set([str(r) for r in range(beat_range[0], beat_range[1])])
    pitches = set(
        [
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )

    notes_before_removal = len(e)

    # remove if in beat range and pitch range
    e = [
        ev
        for ev in e
        if not (ev.a["onset/beat"].issubset(beats) and ev.a["pitch"].issubset(pitches))
    ]

    notes_after_removal = len(e)

    notes_removed = notes_before_removal - notes_after_removal

    # add empty notes
    # e += [EventConstraint().force_inactive() for _ in range(40)]

    infill_constraint = {
        "pitch": {
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        }
        | {"-"},
        "onset/beat": {str(r) for r in range(beat_range[0], beat_range[1])} | {"-"},
        "offset/beat": {str(r) for r in range(beat_range[0], beat_range[1])} | {"-"},
        "instrument": ({"Drums"} if drums else ALL_INSTRUMENTS - {"Drums"}) | {"-"},
        "tag": {tag, "-"},
        "tempo": {str(quantize_tempo(tempo)), "-"},
    }

    # count notes per beat

    # add between notes_to_remove - 10 and notes_to_remove + 10 notes. At least 10 notes
    # lower_bound_notes = max(notes_removed - 10, 10)
    # upper_bound_notes = notes_removed + 10
    # add between 0 and 
    # add 3 forced active
    e += [
        EventConstraint().intersect(infill_constraint).force_active()
        for _ in range(3)
    ]
    if notes_removed > 0:
        e += [EventConstraint().intersect(infill_constraint).force_active() for _ in range(notes_removed//2)]
        e += [EventConstraint().intersect(infill_constraint) for _ in range(notes_removed)]
    # 

    print(f"Notes removed: {notes_removed}")

    # # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # add 10 empty notes
    # e += [EventConstraint().force_inactive() for _ in range(40)]

    return e


def repitch(e, beat_range, pitch_range, drums, tag="other", tempo=120):
    print(drums)
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(quantize_tempo(tempo))}

    beats = set([str(r) for r in range(beat_range[0], beat_range[1])])
    pitches = set(
        [
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/beat"].issubset(beats) and e[i].a["pitch"].issubset(pitches):
            e[i].a["pitch"] = pitches
    # pad with empty notes
    e += [EventConstraint().force_inactive() for e in range(N_EVENTS - len(e))]
    return e

def revelocity(e, beat_range, pitch_range, drums, tag="other", tempo=120):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(quantize_tempo(tempo))}

    beats = set([str(r) for r in range(beat_range[0], beat_range[1])])
    pitches = set(
        [
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/beat"].issubset(beats) and e[i].a["pitch"].issubset(pitches):
            e[i].a["velocity"] = ALL_VELOCITIES
    # pad with empty notes
    e += [EventConstraint().force_inactive() for e in range(N_EVENTS - len(e))]
    return e

def retime(e, beat_range, pitch_range, drums, tag="other", tempo=120):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(quantize_tempo(tempo))}

    beats = set([str(r) for r in range(beat_range[0], beat_range[1])])
    pitches = set(
        [
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/beat"].issubset(beats) and e[i].a["pitch"].issubset(pitches):
            e[i].a["onset/beat"] = beats
            e[i].a["offset/beat"] = beats
            e[i].a["onset/tick"] = ALL_ONSET_TICKS
            e[i].a["offset/tick"] = ALL_OFFSET_TICKS

    # pad with empty notes
    e += [EventConstraint().force_inactive() for e in range(N_EVENTS - len(e))]
    return e

def reinstrument(e, beat_range, pitch_range, drums, tag="other", tempo=120):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(quantize_tempo(tempo))}

    beats = set([str(r) for r in range(beat_range[0], beat_range[1])])
    pitches = set(
        [
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/beat"].issubset(beats) and e[i].a["pitch"].issubset(pitches):
            e[i].a["instrument"] = {"Drums"} if drums else ALL_INSTRUMENTS - {"Drums"}

    # pad with empty notes
    e += [EventConstraint().force_inactive() for e in range(N_EVENTS - len(e))]
    return e


@app.route("/replace", methods=["POST"])
def replace():
    data = request.json
    # recursively convert keys from camelCase to snake_case
    data = json_camel_to_snake(data)
    action = data["action"]
    sampling_settings = data["sampling_settings"] 
    try:
        pitch_range = data["replace_info"]["pitch_range"]
        tick_range = data["replace_info"]["tick_range"]
        sequence = data["harm_seq"] + data["drum_seq"]
        tempo = data["tempo"]
        edit_drums = data["replace_info"]["replace_part"] == "drum_seq"
        e = seq2events(sequence, tempo)

        beat_range = [ int(tick_range[0]) // model.tokenizer.config["ticks_per_beat"], 1+int(tick_range[1]) // model.tokenizer.config["ticks_per_beat"]]
        pitch_range = [int(pitch_range[0]), int(pitch_range[1])]
        if action == "replace":
            e = infill(
                e, beat_range, pitch_range, drums=edit_drums, tag="pop", tempo=tempo
            )
        elif action == "repitch":
            e = repitch(e, beat_range, pitch_range, drums=edit_drums, tag="pop", tempo=tempo)
        elif action == "retime":
            e = retime(e, beat_range, pitch_range, drums=edit_drums, tag="pop", tempo=tempo)
        elif action == "reinstrument":
            e = reinstrument(e, beat_range, pitch_range, drums=edit_drums, tag="pop", tempo=tempo)
        elif action == "revelocity":
            e = revelocity(e, beat_range, pitch_range, drums=edit_drums, tag="pop", tempo=tempo)
            print(e)
        mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)

        x = generate(
            mask,
            top_p=float(sampling_settings["topp"]),
            temperature=float(sampling_settings["temperature"]),
            top_k=int(sampling_settings["topk"]),
            tokens_per_step=int(sampling_settings["greed"]),
            order="random",
        )
        x_sm = model.tokenizer.decode(x)
        x_sm = util.sm_fix_overlap_notes(x_sm)

        # print note num
        print(f"Note num: {x_sm.note_num()}")

        loop_object = sm_to_looprep(x_sm)

        return jsonify(loop_object)
    except Exception as e:
        import traceback

        etype, value, tb = sys.exc_info()
        print(traceback.print_exception(etype, value, tb))
        print(e)
        return jsonify({"error": str(e)}), 500



def add_snare_ghost_notes(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    e += [
        EventConstraint()
        .intersect({"pitch": {"38 (Drums)", "-"}} | velocity_constraint(50))
        .force_active()
        for _ in range(5)
    ]

    e += [
        EventConstraint().intersect(
            {"pitch": {"38 (Drums)", "-"}} | velocity_constraint(80)
        )
        for _ in range(20)
    ]
    # pad
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    return e

def add_percussion(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # remove percussion
    e = [ev for ev in e if ev.a["pitch"].isdisjoint(PERCUSSION_PITCHES)]

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

def add_tom_fill(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]
    # remove drums in last 4 bars
    e = [
        ev
        for ev in e
        if not (
            not ev.a["onset/beat"].isdisjoint({ "14", "15"})
            and not ev.a["instrument"].isdisjoint({"Drums"})
        )
    ]
    # add 3 toms from any of the tom pitches.
    e += [
        EventConstraint()
        .intersect(
            {
                "instrument": {"Drums"},
                "pitch": TOM_PITCHES,
                "onset/beat": {"14", "15", "_"},
            }
        )
        .force_active()
        for e in range(3)
    ]
    # add up to 10 more drums
    e += [
        EventConstraint().intersect(
            {"instrument": {"Drums"}, "onset/beat": {"14", "15", "_"}}
        )
        for _ in range(10)
    ]
    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e


def add_dynamic_hihats(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]
    # remove hihats
    e = [ev for ev in e if ev.a["pitch"].isdisjoint(HIHAT_PITCHES)]
    e += [
        EventConstraint()
        .intersect({"pitch": HIHAT_PITCHES} | velocity_constraint(30))
        .force_active()
        for _ in range(3)
    ]

    e += [
        EventConstraint()
        .intersect({"pitch": HIHAT_PITCHES} | velocity_constraint(60))
        .force_active()
        for _ in range(3)
    ]

    # add up to 10 more
    e += [
        EventConstraint().intersect({"pitch": HIHAT_PITCHES | {"-"}}) for _ in range(N_EVENTS - len(e))]

    # pad
    # e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    return e

def add_arpeggio(e):

    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # remove all sytnh lead
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Synth Lead"})]
    
    # add synth 
    # add 40 synth lead, 4 per beat
    for beat in range(16):
        for tick in range(4):
            e += [
                EventConstraint()
                .intersect({"instrument": {"Synth Lead"}})
                .intersect(
                    {
                        "onset/beat": {str(beat)},
                        "onset/tick": {
                            str(model.tokenizer.config["ticks_per_beat"] // 4 * tick)
                        },
                        "pitch": {str(note) for note in range(50, 100)},
                    }
                )
                .force_active()
            ]

    # add 2 forced synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Synth Lead"}}).force_active()
        for _ in range(2)
    ]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e


def add_lead(e):    

    tag = "pop"
    instrument = "Synth Lead"

    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # set all tags to jazz
    for i in range(len(e)):
        e[i].a["tag"] = {tag}

    # remove piano
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({instrument})]

    e += [
        EventConstraint()
        .intersect({"instrument": {instrument}, "pitch": {str(note) for note in range(55, 100)}})
        .force_active()
    ]

    # add optional Brass notes 
    e += [
        EventConstraint().intersect(
            {
                "instrument": {instrument, "-"},
                "pitch": {str(note) for note in range(40, 100)} | {"-"},
            }
        )
        for _ in range(20)
    ]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]   

    # add tag constraint
    e = [ev.intersect({"tag":{tag,"-"}}) for ev in e]

    return e

# add some chords
def add_chords(e):
    tag = "pop"

    instrument = "Piano"

    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # set all tags to jazz
    for i in range(len(e)):
        e[i].a["tag"] = {tag}

    # remove piano
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({instrument})]

    # add a 3 Guitar notes on first beat
    e += [
        EventConstraint()
        .intersect(
            {
                "instrument": {instrument},
                "onset/beat": {"0"},
                "offset/beat": {"2", "3", "4"},
            } #   | scale_constraint("C major", (50,100))
        )
        .force_active()
        for i in range(3)
    ]

    # add optionalinstrumentnotes
    e += [EventConstraint().intersect({"instrument": {instrument, ""}}) for _ in range(30)]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    # add tag constraint
    e = [ev.intersect({"tag":{tag,"-"}}) for ev in e]

    return e

# add bassline that matches kick
def add_locked_in_bassline(e):
    # remove inactive notes
    e = [ev for ev in e if ev.is_active()]
    # remove bass
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
    # find kicks
    kicks = [
        ev
        for ev in e
        if {"35 (Drums)", "36 (Drums)", "37 (Drums)"}.intersection(ev.a["pitch"])
    ]
    # add bass on every kick
    for kick in kicks:
        e += [
            EventConstraint()
            .intersect(
                {
                    "instrument": {"Bass"},
                    "onset/beat": kick.a["onset/beat"],
                    "onset/tick": kick.a["onset/tick"],
                }
            )
            .force_active()
        ]
    # add up to 5 more bass notes
    e += [EventConstraint().intersect({"instrument": {"Bass"}}) for _ in range(5)]
    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e


@app.route("/custom", methods=["POST"])
def custom():
    data = request.json
    # recursively convert keys from camelCase to snake_case
    data = json_camel_to_snake(data)
    action = data["action"]
    sampling_settings = data["sampling_settings"]
    try:
        pitch_range = data["replace_info"]["pitch_range"]
        tick_range = data["replace_info"]["tick_range"]
        sequence = data["harm_seq"] + data["drum_seq"]
        tempo = data["tempo"]
        edit_drums = data["replace_info"]["replace_part"] == "drum_seq"
        e = seq2events(sequence, tempo)

        beat_range = [
            int(tick_range[0]) // model.tokenizer.config["ticks_per_beat"],
            1 + int(tick_range[1]) // model.tokenizer.config["ticks_per_beat"],
        ]
        pitch_range = [int(pitch_range[0]), int(pitch_range[1])]
        if action == "tom_fill":
            e = add_tom_fill(e)
        elif action == "snare_ghost_notes":
            e = add_snare_ghost_notes(e)
        elif action == "percussion":
            e = add_percussion(e)
        elif action == "dynamic_hihats":
            e = add_dynamic_hihats(e)
        elif action == "locked_in_bassline":
            e = add_locked_in_bassline(e)
        elif action == "chords":
            e = add_chords(e)
        elif action == "lead":
            e = add_lead(e)
        elif action == "arpeggio":
            e = add_arpeggio(e)
        else:
            raise ValueError(f"Unknown action: {action}")
         
        mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)

        x = generate(
            mask,
            top_p=float(sampling_settings["topp"]),
            temperature=float(sampling_settings["temperature"]),
            top_k=int(sampling_settings["topk"]),
            tokens_per_step=int(sampling_settings["greed"]),
            order="random",
        )
        x_sm = model.tokenizer.decode(x)
        x_sm = util.sm_fix_overlap_notes(x_sm)

        # print note num
        print(f"Note num: {x_sm.note_num()}")

        loop_object = sm_to_looprep(x_sm)

        return jsonify(loop_object)
    except Exception as e:
        import traceback

        etype, value, tb = sys.exc_info()
        print(traceback.print_exception(etype, value, tb))
        print(e)
        return jsonify({"error": str(e)}), 500



@app.route("/create", methods=["POST"])
def create():
    data = request.json
    # recursively convert keys from camelCase to snake_case
    data = json_camel_to_snake(data)
    sampling_settings = data["sampling_settings"]
    action = data["action"]
    try:
        if action == "disco":
            e = disco_beat()
        elif action == "metal":
            e = metal_beat()
        elif action == "goofy":
            e = fun_beat()
        elif action == "synth":
            e = synth_beat()
        elif action == "breakbeat":
            e = breakbeat()
        elif action == "funk":
            e = funk_beat()
        else:
            raise ValueError(f"Unknown action: {action}")

            
        # e = breakbeat()
        # e = four_on_the_floor_beat()
        # e = disco_beat()
        # e = metal_beat()
        # e = fun_beat()
        # e = synth_beat()
        # e = simple_beat()
        mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
 
        x = generate(
            mask,
            top_p= float(sampling_settings["topp"]),
            temperature=float(sampling_settings["temperature"]),
            top_k = int(sampling_settings["topk"]),
            tokens_per_step = int(sampling_settings["greed"]) ,
            order="random")
        x_sm = model.tokenizer.decode(x)
        x_sm = util.sm_fix_overlap_notes(x_sm)
        
        # print note num
        print(f"Note num: {x_sm.note_num()}")
    
        loop_object = sm_to_looprep(x_sm)

        return jsonify(loop_object)
    except Exception as ex:
        import traceback
        etype, value, tb = sys.exc_info()
        print(traceback.print_exception(etype, value, tb))
        print(ex)
        return jsonify({"error": str(ex)}), 500
# %%
