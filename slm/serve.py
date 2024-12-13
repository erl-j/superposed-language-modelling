import sys
import socket
import os
import random
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
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
from conversion_utils import looprep_to_sm, sm_to_events, sm_to_looprep

USE_FP16 = False


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cuda:5"
LLM_DEVICE = "cuda:4"
    
USE_LOCAL_LLM = False

ROOT_DIR = "./"

MODEL = "slm"

OUTPUT_DIR = ROOT_DIR + "artefacts/examples_4"
TMP_DIR = ROOT_DIR + "artefacts/tmp"

os.makedirs(OUTPUT_DIR, exist_ok=True)



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

model = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + checkpoints[MODEL],
        map_location=device,
    )
    .to(device)
    .eval()
)

model = SuperposedLanguageModel.load_from_checkpoint(
    # "./checkpoints/zesty-dawn-376/last.ckpt",
    # "./checkpoints/faithful-wave-417/last.ckpt",
    # "./checkpoints/vibrant-paper-422/last.ckpt",
    # "./checkpoints/desert-dust-401/last.ckpt",
    # "./checkpoints/smart-wood-419/last.ckpt",
    # "./checkpoints/unique-tree-426/last.ckpt",
    # "./checkpoints/bumbling-dream-427/last.ckpt",
    # "./checkpoints/lively-flower-428/last.ckpt",
    # "./checkpoints/sparkling-dust-435/last.ckpt",
    # "./checkpoints/pretty-smoke-437/last.ckpt",
    # "./checkpoints/desert-dragon-439/last.ckpt",
    # "./checkpoints/efficient-flower-443/last.ckpt",
    # "./checkpoints/sparkling-dust-435/epoch=271-step=1175856-val/loss_epoch=0.16102.ckpt",
    # "./checkpoints/misunderstood-eon-449/last.ckpt",
    # "./checkpoints/chocolate-river-450/last.ckpt",
    # "./checkpoints/fragrant-dew-452/last.ckpt",
    # "./checkpoints/lilac-feather-455/last.ckpt",
    # "./checkpoints/copper-monkey-456/last.ckpt",
    # "./checkpoints/fragrant-dew-452/last.ckpt",
    # "./checkpoints/ruby-glade-461/last.ckpt",
    # "./checkpoints/prime-cosmos-462/last.ckpt",
    # "./checkpoints/drawn-universe-463/last.ckpt",
    # "./checkpoints/dulcet-jazz-464/last.ckpt",
    # "./checkpoints/clean-oath-465/last.ckpt",
    "./checkpoints/stoic-capybara-480/last.ckpt",
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
ec = lambda: MusicalEventConstraint(model.tokenizer)

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


def generate_function(prompt):

    system_prompt = f"""
    Your task is to write a function that creates a constraint for creating or editing a MIDI loop according to a user provided prompt.
    The constraint is fed to a state of the art machine learning model that generates the loop from the constraints.
    The attributes of each note event are: {model.tokenizer.note_attribute_order}.
    Each attribute is constrained to a set of string values.
    The tag attribute can take on the values : {model.tokenizer.config['tags']}.
    The instruments that are available are the "Piano", "Guitar", "Bass" and "Drums".
    Pitch is expressed in MIDI pitch number. Drums have their own pitch values, in the form of "pitch (Drums)".
    To set a pitch constraint, intersect the event constraint with a set of pitch values.
    Onset and offset are specified in beats (quarters) (integer 0-15) and ticks (integer 0-23) with 24 ticks per beat.
    Two special attributes are "tempo" and "velocity" which are set only using tempo_constraint and velocity_constraint methods. 
    These two are the only attributes with specific constraint methods. The rest of the attributes are set by intersecting the event with a dictionary of attribute values.
    To set velocity, use the velocity_constraint method. For example, ev.intersect(ec().velocity_constraint(40)) sets the velocity of the event to near 40.
    To set tempo, use the tempo_constraint method. For example, ev.intersect(ec().tempo_constraint(120)) sets the tempo of the event to near 120.
    Each request should be considered independently of the previous requests. Make sure that the constraint is just detailed enough to generate the desired loop.
    Do not try to generate the loop directly in the function, but rather create a constraint that can be used to generate the loop.
    Always preserve tempo and tag information from the input events.
    Do not invent any new methods not provided in the examples.
    """.strip()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Example 1: Generate a funk beat with snares on the 2 and 4 and triplets in the last bar!",
                }
            ],
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
                                # set aside other instruments
                                e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                                # add 10 kicks
                                e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(10)] # pitch is expressed in MIDI pitch nr, note that only drums have their pitch in the form of "pitch (Drums)"
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
                                    .intersect(ec().velocity_constraint(40)) # velocity constraint takes one argument
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
                                # add back other instruments
                                e += e_other
                                # important: always pad with empty notes!
                                e += [ec().force_inactive() for _ in range(n_events - len(e))]

                                return e  # return the events""".strip(),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Example 2: Add a bassline that locks in with the kick drum.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": """
                            def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                                # remove inactive notes
                                e = [ev for ev in e if ev.is_active()]
                                # set aside other instruments
                                e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                                # start over
                                e = []
                                # find kicks
                                kicks = [
                                    ev
                                    for ev in e_other
                                    if {"35 (Drums)", "36 (Drums)", "37 (Drums)"}.intersection(ev.a["pitch"])
                                ]
                                # add bass on every kick
                                for kick in kicks:
                                    e += [
                                        ec()
                                        .intersect(
                                            {
                                                "instrument": {"Bass"},
                                                "onset/beat": kick.a["onset/beat"],
                                                "onset/tick": kick.a["onset/tick"],
                                                "pitch": {str(pitch) for pitch in range(30, 46)}, # set pitch range to low pitches
                                            }
                                        )
                                        .force_active()
                                    ]
                                # add up to 5 more bass notes
                                e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(5)]
                                # intersect with current tempo
                                e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                                # set tag to current tag
                                e = [ev.intersect({"tag": {tag}}) for ev in e]

                                # add back other instruments
                                e += e_other
                                # pad with empty notes
                                e += [ec().force_inactive() for _ in range(n_events - len(e))]
                                return e""".strip(),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Here is a new beat." + prompt}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": """
                        def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to""".strip(),
                },
            ],
        },
    ]

    if USE_LOCAL_LLM:

        if pipe is None:
            pipe = pipeline(
                "text-generation",
                model="meta-llama/Llama-3.2-3B-Instruct",
                device=LLM_DEVICE,
                torch_dtype=torch.bfloat16,
                temperature=0.01,
            )

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
            max_tokens=2000,
            temperature=0,
            system=system_prompt,
            messages=messages,
        )
        # combine last two assistant messages
        fn = messages[-1]["content"][0]["text"] + response.content[0].text
    return fn

llm_ram_cache = {}

pipe = None

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
        tempo = data["tempo"]
        edit_drums = data["replace_info"]["replace_part"] == "drum_seq"

        print(f"got pitch range: {pitch_range}")
        midi = looprep_to_sm(data, model.tokenizer.config["ticks_per_beat"])

        DEFAULT_TAG = "pop"

        e = sm_to_events(midi,tag=DEFAULT_TAG, tokenizer=model.tokenizer)
        tick_range = [
            int(tick_range[0]),
            1 + int(tick_range[1]),
        ]
        print(tick_range)
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
        elif action == "unconditional":
            e = unconditional(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "replace":
            e = replace(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "repitch":
            e = repitch(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "retime":
            e = retime(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "reinstrument":
            e = reinstrument(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "revelocity":
            e = revelocity(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "disco":
            # e = disco_beat(
            #     e,
            #     ec,
            #     n_events,
            #     tick_range,
            #     pitch_range,
            #     drums=edit_drums,
            #     tag=DEFAULT_TAG,,
            #     tempo=tempo,
            # )
            e = band_beat(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "metal":
            e = metal_beat(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "goofy":
            e = fun_beat(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "synth":
             e = band_beat(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
            # e = synth_beat(
            #     e,
            #     ec,
            #     n_events,
            #     beat_range,
            #     pitch_range,
            #     drums=edit_drums,
            #     tag=DEFAULT_TAG,,
            #     tempo=tempo,
            # )
        elif action == "breakbeat":
            # e = breakbeat(
            #     e,
            #     ec,
            #     n_events,
            #     beat_range,
            #     pitch_range,
            #     drums=edit_drums,
            #     tag=DEFAULT_TAG,,
            #     tempo=tempo,
            # )
            e = reggaeton_beat(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "funk":
            e = funk_beat(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "tom_fill":
            e = add_tom_fill(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "snare_ghost_notes":
            e = add_snare_ghost_notes(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "percussion":
            e = add_percussion(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "dynamic_hihats":
            e = add_dynamic_hihats(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "locked_in_bassline":
            e = add_locked_in_bassline(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "chords":
            e = add_chords(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "lead":
            e = add_lead(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "arpeggio":
            e = add_arpeggio(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "funky_bassline":
            e = funky_bassline(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "humanize":
            e = humanize(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        elif action == "jazz piano":
            e = jazz_piano(
                e,
                ec,
                n_events,
                tick_range,
                pitch_range,
                drums=edit_drums,
                tag=DEFAULT_TAG,
                tempo=tempo,
            )
        else:
            raise ValueError(f"Unknown action: {action}")
        
        # if more than n_events, warn!
        if len(e) > n_events:
            print(f"Warning: More than {n_events} events generated: {len(e)}")
            # take random subset of 
            e = random.sample(e, n_events)

        mask = model.tokenizer.event_constraints_to_mask(e).to(device)

        # mask = model.fast_kill_events(mask)

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


        def count_less_than_1_durations(x_sm):
            negative_durations = 0
            for track in x_sm.tracks:
                for note in track.notes:
                    if note.duration < 0:
                        negative_durations += 1
            return negative_durations
        # print count notes with negative durations
        print(f"Numb of notes with negative durations: {count_less_than_1_durations(x_sm)}")

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
