import os
import torch
import random
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import time
from train import TrainingWrapper
from data import MidiDataset
from conversion_utils import sm_to_events
from constraints.core import MusicalEventConstraint
from paper_checkpoints import CHECKPOINTS
from util import sm_set_track_order, sm_fix_overlap_notes

# Number of examples to generate per task
N_EXAMPLES = 250
GENERATE = True
ORDER = "random"

OUTPUT_DIR = Path("./artefacts/applications_250e")

# replace given instrument set
def replace_w_instrument_set(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):

    # get tempo
    tempo = None
    for ev in e:
        if ev.is_active():
            tempo = ev.a["tempo"]
            break
    tag = None
    for ev in e:
        if ev.is_active():
            tag = ev.a["tag"]
            break
    
    instruments = set()
    # get instrument tokens currently present in the events
    for ev in e:
        if ev.is_active():
            instruments.update(ev.a["instrument"])

    # count number of active events
    n_active_events = sum([1 for ev in e if ev.is_active()])

    # force one note per instrument
    instrument_list = list(instruments)
    e = [ec().intersect({"instrument": {instrument_list[i]}}).force_active() for i in range(len(instrument_list))]

    n_free_instrument_to_add = n_active_events - len(e)

    e += [ec().intersect({"instrument": instruments}).force_active() for _ in range(n_free_instrument_to_add)]

    # set tempo to tempo and tag to tag
    e = [ev.intersect({"tempo": tempo, "tag": tag}) for ev in e]

    # pad with inactive events
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    print(e)

    return e


def bass_and_drums(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
    e = []

    # force 8 bass note
    e += [ec().intersect({"instrument": {"Bass"}}).force_active() for _ in range(8)]
    # force 1 drums note
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for _ in range(8)]

    # constrain instrument to be only bass and drums
    e += [ec().intersect({"instrument": {"Bass", "Drums"}}).force_active() for i in range(64)]

    # add 64 optional bass and drums
    e += [ec().intersect({"instrument": {"Bass", "Drums", "-"}}) for i in range(64)]
    # pad
    # set tag to pop
    e = [ev.intersect({"tag": {"rock", "-"}}) for ev in e]
    # set tempo to 130
    e = [ev.intersect(ec().tempo_constraint(130)) for ev in e]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

def pipe_and_chromatic_percussion(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
    e = []

    # force 8 strings notes
    e += [ec().intersect({"instrument": {"Chromatic Percussion"}}).force_active() for _ in range(8)]

    # force 8 flute notes
    e += [ec().intersect({"instrument": {"Pipe"}}).force_active() for _ in range(8)]

    # add 80 optional Chromatic Percussion and flute notes
    e += [ec().intersect({"instrument": {"Pipe", "Chromatic Percussion"}}) for _ in range(80)]

    e = [ev.intersect({"tag": {"classical", "-"}}) for ev in e]

    # set tempo to 130
    e = [ev.intersect(ec().tempo_constraint(130)) for ev in e]
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

def piano_pentatonic(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):

    e = []
    # add 8 optional piano notes in C major pentatonic pitch set
    e += [ec().intersect({"instrument": {"Piano"}}).intersect(ec().pitch_in_scale_constraint("C pentatonic", (36,108))).force_active() for i in range(8) ]

    # add 64 optional piano notes in C major pentatonic pitch set
    e += [ec().intersect({"instrument": {"Piano"}}).intersect(ec().pitch_in_scale_constraint("C pentatonic", (36,108))) for i in range(64)]

    # pad with inactive events
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    # set tempo
    e = [ev.intersect(ec().tempo_constraint(130)) for ev in e]

    return e

def constrained_generation(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
        
        n_active_events = sum([1 for ev in e if ev.is_active()])
        e = [ec().intersect(
            {
            "instrument": set().union(*[ev.a["instrument"] for ev in e if ev.is_active()]),
            "pitch": set().union(*[ev.a["pitch"] for ev in e if ev.is_active()]),
            # "onset/global_tick": set().union(*[ev.a["onset/global_tick"] for ev in e if ev.is_active()]),
            "duration": set().union(*[ev.a["duration"] for ev in e if ev.is_active()]),
            "velocity": set().union(*[ev.a["velocity"] for ev in e if ev.is_active()]),
            "tempo": set().union(*[ev.a["tempo"] for ev in e if ev.is_active()]),
            "tag": set().union(*[ev.a["tag"] for ev in e if ev.is_active()])
        }).force_active()
        for _ in range(n_active_events)]

        # pad with inactive events
        e += [ec().force_inactive() for _ in range(n_events - len(e))]
        return e

def constrained_half(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):

    n_active_events = sum([1 for ev in e if ev.is_active()])

    e = [ec().intersect(
        {
        "instrument": set().union(*[ev.a["instrument"] for ev in e if ev.is_active()]),
        "pitch": set().union(*[ev.a["pitch"] for ev in e if ev.is_active()]),
        # "onset/global_tick": set().union(*[ev.a["onset/global_tick"] for ev in e if ev.is_active()]),
        "duration": set().union(*[ev.a["duration"] for ev in e if ev.is_active()]),
        "velocity": set().union(*[ev.a["velocity"] for ev in e if ev.is_active()]),
        "tempo": set().union(*[ev.a["tempo"] for ev in e if ev.is_active()]),
        "tag": set().union(*[ev.a["tag"] for ev in e if ev.is_active()])
    }).force_active()
    for _ in range(n_active_events//2)]

    unconstrained_events_to_add = n_active_events - len(e)

    e += [ec().force_active() for _ in range(unconstrained_events_to_add)]

    # pad with inactive events
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

def constrained_quarter(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
    n_active_events = sum([1 for ev in e if ev.is_active()])

    e = [
        ec()
        .intersect(
            {
                "instrument": set().union(
                    *[ev.a["instrument"] for ev in e if ev.is_active()]
                ),
                "pitch": set().union(*[ev.a["pitch"] for ev in e if ev.is_active()]),
                # "onset/global_tick": set().union(*[ev.a["onset/global_tick"] for ev in e if ev.is_active()]),
                "duration": set().union(
                    *[ev.a["duration"] for ev in e if ev.is_active()]
                ),
                "velocity": set().union(
                    *[ev.a["velocity"] for ev in e if ev.is_active()]
                ),
                "tempo": set().union(*[ev.a["tempo"] for ev in e if ev.is_active()]),
                "tag": set().union(*[ev.a["tag"] for ev in e if ev.is_active()]),
            }
        )
        .force_active()
        for _ in range(n_active_events // 4)
    ]

    unconstrained_events_to_add = n_active_events - len(e)

    e += [ec().force_active() for _ in range(unconstrained_events_to_add)]

    # pad with inactive events
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

def constrained_eight(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
    n_active_events = sum([1 for ev in e if ev.is_active()])

    e = [
        ec()
        .intersect(
            {
                "instrument": set().union(
                    *[ev.a["instrument"] for ev in e if ev.is_active()]
                ),
                "pitch": set().union(*[ev.a["pitch"] for ev in e if ev.is_active()]),
                # "onset/global_tick": set().union(*[ev.a["onset/global_tick"] for ev in e if ev.is_active()]),
                "duration": set().union(
                    *[ev.a["duration"] for ev in e if ev.is_active()]
                ),
                "velocity": set().union(
                    *[ev.a["velocity"] for ev in e if ev.is_active()]
                ),
                "tempo": set().union(*[ev.a["tempo"] for ev in e if ev.is_active()]),
                "tag": set().union(*[ev.a["tag"] for ev in e if ev.is_active()]),
            }
        )
        .force_active()
        for _ in range(n_active_events // 8)
    ]

    unconstrained_events_to_add = n_active_events - len(e)

    e += [ec().force_active() for _ in range(unconstrained_events_to_add)]

    # pad with inactive events
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e


def replace_pitches_given_instrument_pitch_set(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
    # get all instrument tokens currently present in the events
    current_instruments = set()
    for ev in e:
        if ev.is_active():
            current_instruments.update(ev.a["instrument"])

    instruments_to_replace = set(random.sample(current_instruments, min(2, len(current_instruments))))

    # for each instrument, get the pitch set of that instrument
    instrument_pitch_set = {}
    for event_idx in range(n_events):
        if e[event_idx].is_active():
            for instrument in e[event_idx].a["instrument"]:
                if instrument not in instrument_pitch_set:
                    instrument_pitch_set[instrument] = set()
                instrument_pitch_set[instrument].update(e[event_idx].a["pitch"])
    
    # for events whose instrument is in instruments_to_replace, replace pitch with the pitch set of instrument
    for event_idx in range(n_events):
        if e[event_idx].is_active() and len(e[event_idx].a["instrument"].intersection(instruments_to_replace)) > 0:
            # get instrument token
            instrument = e[event_idx].a["instrument"].pop()
            e[event_idx].a["pitch"] = instrument_pitch_set[instrument]
    return e

def replace_pitches_given_pitch_set(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):

    # current tempo
    current_tempos = set()
    for ev in e:
        current_tempos.update(ev.a["tempo"])

    # current tags
    current_tags = set()
    for ev in e:
        current_tags.update(ev.a["tag"])


    # get union of pitches of all notes
    current_pitches = set()
    for ev in e:
        if ev.is_active() and ev.a["instrument"] != {"Drums"}:
            current_pitches.update(ev.a["pitch"])

    # for current active pitches. replace pitch
    for event_idx in range(n_events):
        if e[event_idx].is_active() and e[event_idx].a["instrument"] != {"Drums"}:
            e[event_idx].a["pitch"] = current_pitches

    # constrain to current tempo
    for event_idx in range(n_events):
        e[event_idx] = e[event_idx].intersect({"tempo": current_tempos})
    # constrain to current tag
    for event_idx in range(n_events):
        e[event_idx] = e[event_idx].intersect({"tag": current_tags})
    return e

def replace_pitches_given_pitch_set_tiled_across_octaves(
    e, ec, n_events, tick_range, pitch_range, drums, tag, tempo
):
    # current tempo
    current_tempos = set()
    for ev in e:
        current_tempos.update(ev.a["tempo"])

    # current tags
    current_tags = set()
    for ev in e:
        current_tags.update(ev.a["tag"])

    # get union of pitches of all notes
    current_pitches = set()
    for ev in e:
        if ev.is_active() and ev.a["instrument"] != {"Drums"}:
            current_pitches.update(ev.a["pitch"])


    current_pitches_w_octaves = set()
    
    def get_octaves(pitch):
        # get lowest octave
        lowest_octave = int(pitch) % 12

        # get all octaves
        octave_pitches = set()
        for i in range(lowest_octave, 127, 12):
            octave_pitches.add(str(i))
        return octave_pitches

    for pitch in list(current_pitches):
        octaves = get_octaves(pitch)
        for octave in octaves:
            current_pitches_w_octaves.add(str(octave))

    # for current active pitches. replace pitch
    for event_idx in range(n_events):
        if e[event_idx].is_active() and e[event_idx].a["instrument"] != {"Drums"}:
            e[event_idx].a["pitch"] = current_pitches_w_octaves

    # constrain to current tempo
    for event_idx in range(n_events):
        e[event_idx] = e[event_idx].intersect({"tempo": current_tempos})
    # constrain to current tag
    for event_idx in range(n_events):
        e[event_idx] = e[event_idx].intersect({"tag": current_tags})
    return e

def infill_middle(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
    # get tags
    tags = set()
    for ev in e:
        if ev.is_active():
            tags.update(ev.a["tag"])

    # get tempos
    tempos = set()
    for ev in e:
        if ev.is_active():
            tempos.update(ev.a["tempo"])

    # instruments to replace 

    return [
        ec().intersect({
            "pitch": set([f"{r}" for r in range(pitch_range[0], pitch_range[1])]) | {"-"},
            "onset/global_tick": set([str(r) for r in range(tick_range[0], tick_range[1])]) | {"-"},
            "offset/global_tick": set([str(r) for r in range(tick_range[0], tick_range[1])]) | {"-"},
            "tag": tags,
            "tempo": tempos
        }).force_active()
        if (
            len(ev.a["onset/global_tick"].intersection(set([str(r) for r in range(tick_range[0], tick_range[1])]))) > 0
            and len(ev.a["offset/global_tick"].intersection(set([str(r) for r in range(tick_range[0], tick_range[1])]))) > 0
            and len(ev.a["pitch"].intersection(set([f"{r}" for r in range(pitch_range[0], pitch_range[1])]))) > 0
        )
        else ev
        for ev in e
    ]

def replace_2_instruments(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
    # get tags
    tags = set()
    for ev in e:
        if ev.is_active():
            tags.update(ev.a["tag"])

    tempos = set()
    for ev in e:
        if ev.is_active():
            tempos.update(ev.a["tempo"])

    # get all instrument tokens currently present in the events
    current_instruments = set()
    for ev in e:
        if ev.is_active():
            current_instruments.update(ev.a["instrument"])

    # get seed by hashing first element
    seed = hash(tuple(e[0].a["instrument"]))
    # pick 2 instruments from the current instruments with seed as the seed
    random_state = random.Random(seed)

    instruments_to_replace = set(random_state.sample(list(current_instruments), min(2, len(current_instruments))))

    for event_idx in range(n_events):
        if e[event_idx].is_active() and len(e[event_idx].a["instrument"].intersection(instruments_to_replace)) > 0:
            e[event_idx] = ec().force_active()
            e[event_idx]["instrument"] = instruments_to_replace
            e[event_idx] = e[event_idx].intersect({"tag": tags, "tempo": tempos})

    # pad with inactive events
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

def unconditional(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
    return [ec() for _ in range(n_events)]

# Define tasks with their parameters
TASKS = {
    "replace_pitches_given_pitch_set_tiled_across_octaves": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": replace_pitches_given_pitch_set_tiled_across_octaves,
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": None,
        "tempo": None,
    },
    # "bass_and_drums": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": bass_and_drums,
    #     "tick_range": None,
    #     "pitch_range": None,
    #     "drums": None,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "pipe_and_chromatic_percussion": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": pipe_and_chromatic_percussion,
    #     "tick_range": None,
    #     "pitch_range": None,
    #     "drums": None,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "constrained_generation": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": constrained_generation,
    #     "tick_range": None,
    #     "pitch_range": None,
    #     "drums": None,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "constrained_half": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": constrained_half,
    #     "tick_range": None,
    #     "pitch_range": None,
    #     "drums": None,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "constrained_quarter": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": constrained_quarter,
    #     "tick_range": None,
    #     "pitch_range": None,
    #     "drums": None,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "constrained_eight": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": constrained_eight,
    #     "tick_range": None,
    #     "pitch_range": None,
    #     "drums": None,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "replace_pitches_given_pitch_set_2": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": replace_pitches_given_pitch_set,
    #     "tick_range": None,
    #     "pitch_range": None,
    #     "drums": None,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "infill_1_bar_box": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": infill_middle,
    #     "tick_range": (7*24, 11*24),
    #     "pitch_range": (10,125),
    #     "drums": False,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "infill_2_bar_box": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": infill_middle,
    #     "tick_range": (5*24, 13*24),
    #     "pitch_range": (10,125),
    #     "drums": False,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "infill_3_bar_box": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": infill_middle,
    #     "tick_range": (3*24, 15*24),
    #     "pitch_range": (10,125),
    #     "drums": False,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "unconditional": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": unconditional,
    #     "tick_range": None,
    #     "pitch_range": None,
    #     "drums": None,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "replace_2_instruments": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": replace_2_instruments,
    #     "tick_range": None,
    #     "pitch_range": None,
    #     "drums": None,
    #     "tag": None,
    #     "tempo": None,
    # },
    # "replace_w_instrument_set": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": replace_w_instrument_set,
    #     "tick_range": None,
    #     "pitch_range": None,
    #     "drums": None,
    #     "tag": None,
    #     "tempo": None,
    # },
}

# Reverse the order of tasks
# TASKS = dict(reversed(list(TASKS.items())))

def setup_model(checkpoint_path, device):
    """Load and set up the model."""
    print(f"Loading model from {checkpoint_path}...")
    model = TrainingWrapper.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model

def load_test_dataset(tokenizer):
    """Load the test dataset."""
    print("Loading test dataset...")
    mmd_4bar_filter_fn = lambda x: "n_bars=4" in x
    sm_filter_fn = lambda sm: not any(
        track.program == 0 and not track.is_drum and "piano" not in track.name.lower()
        for track in sm.tracks
    )

    test_ds = MidiDataset(
        cache_path="./data/mmd_loops/tst_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn,
        genre_list=tokenizer.config["tags"],
        tokenizer=tokenizer,
        min_notes=16,
        max_notes=tokenizer.config["max_notes"],
        use_random_shift=False,
        sm_filter_fn=sm_filter_fn,
    )
    return test_ds

def main():
    parser = argparse.ArgumentParser(description='Generate MIDI examples for a specific model')
    parser.add_argument('--model', type=str, required=True, 
                      choices=list(CHECKPOINTS.keys()),
                      help='Model checkpoint name to process')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU device ID to use')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed')
    # parser.add_argument('--topp', type=float, default=1)
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the test dataset using first available model's tokenizer
    print("\nInitializing with first available model...")
    first_model = setup_model(next(iter(CHECKPOINTS.values())), device)
    test_dataset = load_test_dataset(first_model.tokenizer)

    # Create and save ground truth examples
    ground_truth_dir = OUTPUT_DIR / "ground_truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    # Sample ground truth examples
    rng = random.Random(args.seed)
    ground_truth_indices = rng.sample(range(len(test_dataset)), N_EXAMPLES)
    ground_truth_examples = [test_dataset[idx] for idx in ground_truth_indices]

    # Save ground truth examples if they don't exist
    if not list(ground_truth_dir.glob("*.mid")):
        print("\nSaving ground truth examples...")
        for i, example in enumerate(ground_truth_examples):
            example_sm = first_model.tokenizer.decode(ground_truth_examples[i]["token_ids"])
            sm_set_track_order(example_sm).dump_midi(ground_truth_dir / f"example_{i}.mid")

    # Load the specified model
    print(f"\nProcessing checkpoint: {args.model}")
    model = setup_model(CHECKPOINTS[args.model], device)
    checkpoint_dir = f"{OUTPUT_DIR}/{args.model}_{ORDER}"

    # Process each task
    for task_name, task_config in TASKS.items():
        records = []
        print(f"\nProcessing task: {task_name}")
        task_dir = Path(checkpoint_dir) / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(ground_truth_indices):
            print(f"Generating example {i+1}/{N_EXAMPLES}")
            
            # Get example from test set
            test_example = ground_truth_examples[i]["token_ids"]
            test_sm = first_model.tokenizer.decode(test_example)
            test_events = sm_to_events(test_sm, "pop", model.tokenizer)
            
            # Apply task function
            tic = time.time()
            task_events = task_config["fn"](
                test_events,
                lambda: MusicalEventConstraint(model.tokenizer),
                model.tokenizer.config["max_notes"],
                task_config["tick_range"],
                task_config["pitch_range"],
                task_config["drums"],
                task_config["tag"],
                task_config["tempo"],
            )
            toc = time.time()
            print(f"Time taken to apply task function: {toc-tic}")

            # Convert to mask and get conditional likelihood
            task_mask = model.tokenizer.event_constraints_to_mask(task_events)
            with torch.no_grad():
                model.eval()
                test_mask = model.tokenizer.event_constraints_to_mask(test_events)
                log_probs = model.model.conditional_log_likelihood(
                    test_mask.to(device), task_mask.to(device)
                )
                print(f"Log probs: {log_probs.item()}")
                
                records.append({
                    "model": args.model,
                    "task": task_name,
                    "log_probs": log_probs.item(),
                    "example": i,
                })
                
                if GENERATE:
                    output_mask = model.generate(
                        task_mask,
                        **task_config["sampling_settings"],
                        order=ORDER,
                    )[0].argmax(dim=-1)
                    
                    try:
                        output_sm = model.tokenizer.decode(output_mask)
                        output_sm = sm_fix_overlap_notes(output_sm)
                        sm_set_track_order(output_sm).dump_midi(task_dir / f"generated_{i}.mid")
                    except Exception as e:
                        print(f"Error decoding MIDI: {e}")

        # Save records for this model
        df = pd.DataFrame(records)
        df.to_csv(f"{str(task_dir)}/records.csv", index=False)

if __name__ == "__main__":
    main()