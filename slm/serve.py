import sys
import socket
import os
import random
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.append("slm/")
from train import EncoderOnlyModel
from util import preview_sm, sm_fix_overlap_notes, loop_sm
import util
from paper_checkpoints import checkpoints
from constraints.addx import *
from constraints.re import *
from constraints.templates import *
from constraints.core import MusicalEventConstraint, DRUM_PITCHES, PERCUSSION_PITCHES, TOM_PITCHES, CRASH_PITCHES, HIHAT_PITCHES

USE_FP16 = True


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cuda:5"
ROOT_DIR = "./"

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
        attribute_temperature=attribute_temperature,
)[0].argmax(axis=1)

def preview(sm):
    sm = sm.copy()
    sm = sm_fix_overlap_notes(sm)
    preview_sm(loop_sm(sm, 4, 4))

if USE_FP16:
    model = model.convert_to_half()

# %%

# all pitches that contain "(Drums)"

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
ec = lambda : MusicalEventConstraint(blank_event_dict, model.tokenizer)

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


def get_external_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    external_ip = s.getsockname()[0]
    s.close()
    return external_ip

print("External IP Address:", get_external_ip())
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return "Hello, World!"

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

def sm_to_looprep(sm):
    # make two sequences. one for drums and one for other instruments
    drum_sequence = []
    harm_sequence = []
    for track in sm.tracks:
        for note in track.notes:
            if track.name == "Drums":
                drum_sequence.append(
                    {
                        "pitch": note.pitch,
                        "onset": note.start,
                        "velocity": note.velocity,
                        "duration": sm.tpq // 4,
                        "instrument": track.name,
                    }
                )
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
        "n_bars": 4,
        "drum_seq": drum_sequence,
        "harm_seq": harm_sequence,
        "ppq": sm.tpq,
    }


def seq2events(sequence, tempo):
    events = []
    for note_event in sequence:
        event = MusicalEventConstraint(
            {
                "pitch": {
                    str(note_event["pitch"]) + " (Drums)"
                    if note_event["instrument"] == "Drums"
                    else str(note_event["pitch"])
                },
                "onset/beat": {
                    str(note_event["onset"] // model.tokenizer.config["ticks_per_beat"])
                },
                "onset/tick": {
                    str(note_event["onset"] % model.tokenizer.config["ticks_per_beat"])
                },
                "velocity": {str(ec().quantize_velocity(int(note_event["velocity"])))},
                "instrument": {note_event["instrument"]},
                "tempo": {str(ec().quantize_tempo(tempo))},
                "tag": {"other"},
                "offset/beat": {
                    str(
                        (note_event["onset"] + note_event["duration"])
                        // model.tokenizer.config["ticks_per_beat"]
                    )
                    if note_event["instrument"] != "Drums"
                    else "none (Drums)"
                },
                "offset/tick": {
                    str(
                        (note_event["onset"] + note_event["duration"])
                        % model.tokenizer.config["ticks_per_beat"]
                    )
                    if note_event["instrument"] != "Drums"
                    else "none (Drums)"
                },
            }, model.tokenizer
        )
        events += [event]
    return events

@app.route("/edit", methods=["POST"])
def edit():
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

        n_events = N_EVENTS
        if action == "replace":
            e = infill(
                e, ec, n_events, beat_range, pitch_range, drums=edit_drums, tag="pop", tempo=tempo
            )
        elif action == "repitch":
            e = repitch(
                e, ec, n_events,  beat_range, pitch_range, drums=edit_drums, tag="pop", tempo=tempo
            )
        elif action == "retime":
            e = retime(
                e, ec, n_events, beat_range, pitch_range, drums=edit_drums, tag="pop", tempo=tempo
            )
        elif action == "reinstrument":
            e = reinstrument(
                e, ec, n_events, beat_range, pitch_range, drums=edit_drums, tag="pop", tempo=tempo
            )
        elif action == "revelocity":
            e = revelocity(
                e, ec, n_events, beat_range, pitch_range, drums=edit_drums, tag="pop", tempo=tempo
            )
        elif action == "disco":
            e = disco_beat(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "metal":
            e = metal_beat(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "goofy":
            e = fun_beat(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "synth":
            e = synth_beat(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "breakbeat":
            e = breakbeat(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "funk":
            e = funk_beat(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "tom_fill":
            e = add_tom_fill(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "snare_ghost_notes":
            e = add_snare_ghost_notes(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "percussion":
            e = add_percussion(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "dynamic_hihats":
            e = add_dynamic_hihats(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "locked_in_bassline":
            e = add_locked_in_bassline(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "chords":
            e = add_chords(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "lead":
            e = add_lead(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "arpeggio":
            e = add_arpeggio(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
        elif action == "funky_bassline":
            e = funky_bassline(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="pop",
                tempo=tempo,
            )
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
# %%
