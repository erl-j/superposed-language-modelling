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

device = "cuda:6"


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

    def generate(mask, temperature=1.0, top_p=1.0, attribute_temperature=None):
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
        ev.intersect(tempo_constraint(160)).intersect({"instrument": {"Drums", "-"}})
        for ev in e
    ]
    return e



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


@app.route("/regenerate", methods=["POST"])
def regenerate():
    try:
        # e = simple_beat()
        e = breakbeat()
        mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
        x = generate(mask, top_p=0.96,temperature=1.0)
        x_sm = model.tokenizer.decode(x)
        xt = model.tokenizer.indices_to_tokens(x)

        # split into events
        n_events = model.tokenizer.config["max_notes"]

        event_tokens = [xt[i : i + len(model.tokenizer.note_attribute_order)] for i in range(0, len(xt), len(model.tokenizer.note_attribute_order))]

        # now turn into event dictionaries
        event_dicts = [
            {key: value for key, value in [token.split(":") for token in event_token]}
            for event_token in event_tokens
        ]

        sequence = []
        for event_dict in event_dicts:
            if event_dict["instrument"] != "-":
                sequence.append({"pitch": event_dict["pitch"].replace(" (Drums)",""), 
                                 "onset": (int(event_dict["onset/beat"]) * int(model.tokenizer.config["ticks_per_beat"]) + int(event_dict["onset/tick"])),
                                 "velocity": event_dict["velocity"],
                                 "duration": 4
                                 })
                tempo = event_dict["tempo"]


        #  const new_song = {
        #   "name": "test",
        #   "version": 0,
        #   "bpm": data.tempo,
        #   "timeSignature": data.time_signature,
        #   "numBars": data.n_bars,
        #   "pitchRange": pitchRange,
        #   "tracks": [
        #     {
        #       "instrument": "drums",
        #       "sequence": data.sequence.map(note =>
        #       ({
        #         "pitch": note.pitch,
        #         "onset": note.onset,
        #         "duration": PPQ / 8,
        #         "velocity": note.velocity
        #       })
        #       )
        #     }
        #   ]
        # }
        # time signature
        loop_object = {
            "time_signature": "4/4",
            "tempo": tempo,
            "n_bars":4,
            "sequence": sequence
        }

        # 

        return jsonify(loop_object)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
# %%
