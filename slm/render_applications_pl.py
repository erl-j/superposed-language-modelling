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
ORDER = "lowest_entropy"

OUTPUT_DIR = Path("./artefacts/applications_250" + (f"_{ORDER}" if ORDER is not "random" else ""))

def bass_and_drums(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
    e = []
    # force 1 bass note
    e += [ec().intersect({"instrument": {"Bass"}}).force_active() for _ in range(3)]
    # force 1 drums note
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for _ in range(3)]

    # constrain instrument to be only bass and drums
    e += [ec().intersect({"instrument": {"Bass", "Drums"}}).force_active() for i in range(50)]

    # add 50 optional bass and drums
    e += [ec().intersect({"instrument": {"Bass", "Drums", "-"}}) for i in range(50)]
    # pad
    # set tag to pop
    e = [ev.intersect({"tag": {"rock", "-"}}) for ev in e]
    # set tempo to 130
    e = [ev.intersect(ec().tempo_constraint(130)) for ev in e]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

def strings_and_flute(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
    e = []

    e += [ec().intersect({"instrument": {"Flute", "Strings"}}).force_active() for _ in range(80)]

    e = [ev.intersect({"tag": {"classical", "-"}}) for ev in e]

    # set tempo to 130
    e = [ev.intersect(ec().tempo_constraint(130)) for ev in e]
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

def replace_pitches_given_pitch_set(
    e,
    ec,
    n_events,
    tick_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # get union of pitches of all notes
    current_pitches = set()
    for ev in e:
        # unless drums or inactive
        if ev.a["instrument"] != {"Drums"} and ev.is_active():
            current_pitches.update(ev.a["pitch"])

    # for current active pitches. replace pitch
    for event_idx in range(n_events):
        if e[event_idx].is_active() and e[event_idx].a["instrument"] != {"Drums"}:
            e[event_idx].a["pitch"] = current_pitches
    return e

# Define tasks with their parameters
TASKS = {
    "bass_and_drums": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": bass_and_drums,
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": None,
        "tempo": None,
    },
    "strings_and_flute": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": strings_and_flute,
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": None,
        "tempo": None,
    },
    "constrained_generation": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": lambda e, ec, n_events, tick_range, pitch_range, drums, tag, tempo: [
            ec().intersect({
                "instrument": set().union(*[ev.a["instrument"] for ev in e if ev.is_active()]),
                "pitch": set().union(*[ev.a["pitch"] for ev in e if ev.is_active()]),
                "onset/global_tick": set().union(*[ev.a["onset/global_tick"] for ev in e if ev.is_active()]),
                # "offset/global_tick": set().union(*[ev.a["offset/global_tick"] for ev in e if ev.is_active()]),
                # "duration": set().union(*[ev.a["duration"] for ev in e if ev.is_active()]),
                "velocity": set().union(*[ev.a["velocity"] for ev in e if ev.is_active()]),
                "tempo": set().union(*[ev.a["tempo"] for ev in e if ev.is_active()]),
                # "tag": set().union(*[ev.a["tag"] for ev in e if ev.is_active()])
            }).force_active()
            for _ in range(n_events)
        ],
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": None,
        "tempo": None,
    },
    # "replace_bass_notes": {
    #     "sampling_settings": {
    #         "temperature": 1.0,
    #         "top_p": 1.0,
    #         "top_k": 0,
    #         "tokens_per_step": 1,
    #     },
    #     "fn": lambda e, ec, n_events, tick_range, pitch_range, drums, tag, tempo: [
    #         ec().intersect({"instrument": {"Bass"}}).force_active()
    #         if ev.a["instrument"] == {"Bass"}
    #         else ev
    #         for ev in e
    #     ],
    #     "tick_range": None,
    #     "pitch_range": None,
    #     "drums": None,
    #     "tag": None,
    #     "tempo": None,
    # },
    "replace_pitches": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": lambda e, ec, n_events, tick_range, pitch_range, drums, tag, tempo: [
            (ec().force_active() if ev.is_active() else ev)
            for ev in e
        ],
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": None,
        "tempo": None,
    },
    "replace_pitches_given_pitch_set_2": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": replace_pitches_given_pitch_set,
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": None,
        "tempo": None,
    },
    
    "infill_middle": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": lambda e, ec, n_events, tick_range, pitch_range, drums, tag, tempo: [
            ec().intersect({
                "pitch": set([f"{r}" for r in range(pitch_range[0], pitch_range[1])]) | {"-"},
                "onset/global_tick": set([str(r) for r in range(tick_range[0], tick_range[1])]) | {"-"},
                "offset/global_tick": set([str(r) for r in range(tick_range[0], tick_range[1])]) | {"-"},
            }).force_active()
            if (
                len(ev.a["onset/global_tick"].intersection(set([str(r) for r in range(tick_range[0], tick_range[1])]))) > 0
                and len(ev.a["pitch"].intersection(set([f"{r}" for r in range(pitch_range[0], pitch_range[1])]))) > 0
            )
            else ev
            for ev in e
        ],
        "tick_range": (8*24, 12*24),
        "pitch_range": (22, 110),
        "drums": False,
        "tag": None,
        "tempo": None,
    },
    "unconditional": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": lambda e, ec, n_events, tick_range, pitch_range, drums, tag, tempo: [
            ec() for _ in range(n_events)
        ],
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": None,
        "tempo": None,
    },
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
    checkpoint_dir = OUTPUT_DIR / str(args.model)
    records = []

    # Process each task
    for task_name, task_config in TASKS.items():
        print(f"\nProcessing task: {task_name}")
        task_dir = checkpoint_dir / task_name
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
                    
                    output_sm = model.tokenizer.decode(output_mask)
                    output_sm = sm_fix_overlap_notes(output_sm)
                    sm_set_track_order(output_sm).dump_midi(task_dir / f"generated_{i}.mid")

    # Save records for this model
    df = pd.DataFrame(records)
    df.to_csv(checkpoint_dir / "records.csv")

if __name__ == "__main__":
    main()