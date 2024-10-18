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
from constraints.core import (
    MusicalEventConstraint,
    DRUM_PITCHES,
    PERCUSSION_PITCHES,
    TOM_PITCHES,
    CRASH_PITCHES,
    HIHAT_PITCHES,
)
from transformers import pipeline
import time

USE_FP16 = True


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cuda:5"
LLM_DEVICE = "cuda:4"
    
USE_LOCAL_LLM = False

ROOT_DIR = "./"

MODEL = "drums"

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
    preview_sm(loop_sm(sm, 4, 4))


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
    n_bars = 4
    drum_sequence = []
    harm_sequence = []
    for track in sm.tracks:
        for note in track.notes:
            if track.name == "Drums":
                drum_sequence.append(
                    {
                        "pitch": note.pitch,
                        "onset": note.start % (sm.tpq * n_bars * 4),
                        "velocity": note.velocity,
                        "duration": sm.tpq // 4,
                        "instrument": track.name,
                    }
                )
            else:
                harm_sequence.append(
                    {
                        "pitch": note.pitch,
                        "onset": note.start % (sm.tpq * n_bars * 4),
                        "velocity": note.velocity,
                        "duration": note.duration,
                        "instrument": track.name,
                    }
                )
    return {
        "time_signature": "4/4",
        "tempo": sm.tempos[0].qpm,
        "n_bars": n_bars,
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
            },
            model.tokenizer,
        )
        events += [event]
    return events


# def generate_function(prompt):
#     function_str = '''def myfun(
#         e,
#         ec,
#         n_events,
#         beat_range,
#         pitch_range,
#         drums,
#         tag,
#         tempo,
#     ):
#         e = [ev for ev in e if not ev.is_inactive()]
#         # add 10 drum notes
#         e += [
#             ec().intersect({"instrument": {"Drums"}}).force_active() for _ in range(10)
#         ]
#         # pad with empty notes
#         e += [ec().force_inactive() for e in range(n_events - len(e))]
#         return e
#     '''
#     return function_str

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B-Instruct",
    device=LLM_DEVICE,
    torch_dtype=torch.bfloat16,
    temperature=0.01,
)

def generate_function(prompt):

    system_prompt = f"""
    Your task is to write a function that creates a constraint for creating a MIDI loop according to a user provided prompt.
    The tag attribute can take on the values : {model.tokenizer.config['tags']}.
    Onset and offset are specified in beats (quarters) (0-15) and ticks (0-23) with 24 ticks per beat.
    """.strip()

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Generate a funk beat with snares on the 2 and 4 and triplets in the last bar!"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": """
                            def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                                '''
                                This funkbeat will contain kicks, snares, hihats (closed and open), ghost snares, and up to 20 optional drum notes.
                                '''
                                e = []
                                # remove all drums
                                e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                                # add 10 kicks
                                e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(10)]
                                # snares on the 2 and 4 of each bar
                                for bar in range(4):
                                    for beat in [1, 3]:
                                        e += [
                                            ec()
                                            .intersect({"pitch": {"38 (Drums)"}})
                                            .intersect({"onset/beat": {str(beat + bar * 4)}})
                                            .force_active()
                                        ]
                                # add 10 hihats
                                e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(10)]
                                # add 4 open
                                e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
                                # add 10 ghost snare
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}})
                                    .intersect(ec().velocity_constraint(40))
                                    .force_active()
                                    for _ in range(10)
                                ]
                                # add up to 20 optional drum notes
                                e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                                # add some triplets in the last bar
                                e += [ec().intersect({"onset/beat": {"12","13","14","15"}, "onset/tick": {"8", "16"}}).force_active() for _ in range(5)]
                                # set tempo to 96
                                e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                                # set tag to funk
                                e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                                # important: always pad with empty notes!
                                e += [ec().force_inactive() for _ in range(n_events - len(e))]
                                return e  # return the events""".strip(),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": """
                        def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a""".strip(),
                },
            ],
        },
    ]

    if USE_LOCAL_LLM:

        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}] + messages

        # process for huggingface pipeline format
        messages = [
            {"role": m["role"], "content": m["content"][0]["text"]} for m in messages
        ]

        response = pipe(
            messages,
            max_new_tokens=1000,
        )

        print(response)

        fn = response[0]["generated_text"][-1]["content"]

    else:
        import anthropic

        client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            # api_key="my_api_key",
        )

        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            system=system_prompt,
            messages=messages,
        )
        # combine last two assistant messages
        fn = messages[-1]["content"][0]["text"] + response.content[0].text
    return fn

llm_ram_cache = {}

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

        if action == "prompt":

            if data["prompt"] in llm_ram_cache:
                func_str = llm_ram_cache[data["prompt"]]
            else:
                func_str = generate_function(data["prompt"])
                # now save both prompt and function to a file
                os.makedirs("llm_cache", exist_ok=True)
                # create filename from timestamp with random hex suffix
                filename = f"llm_cache/{int(time.time())}_{random.randint(0, 1 << 32):x}.py"

                with open(filename, "w") as f:
                    # write the prompt as a comment at the top of the file
                    f.write(f"# {data['prompt']}\n")
                    # now write the function
                    f.write(func_str)

                llm_ram_cache[data["prompt"]] = func_str

            exec(func_str)
            # Extract the function name (assuming it's the first def statement)
            func_name = func_str.split("def ")[1].split("(")[0]
            # Call the function (this assumes no arguments for simplicity)
            e = locals()[func_name](
                e, ec, n_events, beat_range, pitch_range, edit_drums, "other", tempo
            )
        elif action == "replace":
            e = infill(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="other",
                tempo=tempo,
            )
        elif action == "repitch":
            e = repitch(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="other",
                tempo=tempo,
            )
        elif action == "retime":
            e = retime(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="other",
                tempo=tempo,
            )
        elif action == "reinstrument":
            e = reinstrument(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="other",
                tempo=tempo,
            )
        elif action == "revelocity":
            e = revelocity(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="other",
                tempo=tempo,
            )
        elif action == "disco":
            e = disco_beat(
                e,
                ec,
                n_events,
                beat_range,
                pitch_range,
                drums=edit_drums,
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
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
                tag="other",
                tempo=tempo,
            )
        else:
            raise ValueError(f"Unknown action: {action}")
        
        # if more than n_events, warn!
        if len(e) > n_events:
            print(f"Warning: More than {n_events} events generated: {len(e)}")
            # take random subset of 
            e = random.sample(e, n_events)

        mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)

        x = generate(
            mask,
            top_p=float(sampling_settings["topp"]),
            temperature=float(sampling_settings["temperature"]),
            top_k=int(sampling_settings["topk"]),
            tokens_per_step=int(sampling_settings["tokens_per_step"]),
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
