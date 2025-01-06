#%%
import os
import torch
import random
from pathlib import Path
import symusic
from train import TrainingWrapper
from data import MidiDataset
from conversion_utils import looprep_to_sm, sm_to_events, sm_to_looprep
from constraints.core import MusicalEventConstraint
import numpy as np
import math
import time
from paper_checkpoints import CHECKPOINTS
from util import sm_set_track_order, sm_fix_overlap_notes


# Configuration
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
#
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("./artefacts/applications")

# Model checkpoints to test

# if dir already exists, asks for confirmation to delete
if OUTPUT_DIR.exists():
    if input(f"Output directory {OUTPUT_DIR} already exists. Delete? (y/n) ") == "y":
        os.system(f"rm -r {OUTPUT_DIR}")
    else:
        exit()

# Number of examples to generate per task
N_EXAMPLES = 10

GENERATE = True

def triplet_piano_in_scale_pitch_set(e,
    ec,
    n_events,
    tick_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    
        e = []

        # triplet onsets
        onsets = [x for x in range(tick_range[0], tick_range[1], 24//3)]
    
        # add 50 piano notes
        e += [ec().intersect({"instrument": {"Piano"}}).
                intersect(ec().pitch_in_scale_constraint("C major", [35, 88])).
            
        
        # pad with empty notes
        e += [ec().force_inactive() for _ in range(n_events - len(e))]
    
        # set to 96
        e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
    
        return e


def piano_in_scale_pitch_set(e,
    ec,
    n_events,
    tick_range,
    pitch_range,
    drums,
    tag,
    tempo,
):

    e = []

    # add 50 piano notes
    e += [ec().intersect({"instrument": {"Piano"}}).
          intersect(ec().pitch_in_scale_constraint("C major", [35, 88])).force_active() for _ in range(50)]
    
    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    # set to 96
    e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

    return e
    


def constrained_generation(
    e,
    ec,
    n_events,
    tick_range,
    pitch_range,
    drums,
    tag,
    tempo,    
):
    
    # remove inactive notes
    e = [ev for ev in e if ev.is_active()]

    instruments = set().union(*[ev.a["instrument"] for ev in e])
    pitches = set().union(*[ev.a["pitch"] for ev in e])
    onsets = set().union(*[ev.a["onset/global_tick"] for ev in e])
    offsets = set().union(*[ev.a["offset/global_tick"] for ev in e])
    velocities = set().union(*[ev.a["velocity"] for ev in e])
    tempos = set().union(*[ev.a["tempo"] for ev in e])

    # get number of active notes
    n_active_notes = sum([1 for ev in e if ev.is_active()])

    new_events = [
        ec().intersect(
            {"instrument": instruments, "pitch": pitches, "onset/global_tick": onsets, "offset/global_tick": offsets, "velocity": velocities}
        ).intersect({"tempo": tempos}).force_active()
        for _ in range(n_events)
    ]

    return new_events

def replace_bass_notes(
    e,
    ec,
    n_events,
    tick_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # replace bass notes with blank bass notes
    e = [
        ec().intersect({"instrument": {"Bass"}}).force_active()
        if ev.a["instrument"] == {"Bass"}
        else ev
        for ev in e
    ]

    # # add tag constraint
    # e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

    # # add tempo constraint
    # e = [ev.intersect({"tempo": {tempo, "-"}}) for ev in e]

    return e


def replace_notes_in_box(
    e,
    ec,
    n_events,
    tick_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    ticks = set([str(r) for r in range(tick_range[0], tick_range[1])])
    pitches = set(
        [
            f"{r} (Drums)" if drums else f"{r}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )

    valid_onsets = {str(r) for r in range(tick_range[0], tick_range[1])}
    valid_offsets = (
        {"none (Drums)"}
        if drums
        else {str(r) for r in range(tick_range[0] + 4, tick_range[1] + 1)}
    )
    valid_pitches = pitches

    infill_constraint = {
        "pitch": valid_pitches | {"-"},
        "onset/global_tick": ticks | {"-"},
        "offset/global_tick": ticks | {"-"},
        # "duration": valid_durations  | {"-"},
    }

    # replace notes in the middle of the loop
    e = [
        ec().intersect(infill_constraint).force_active()
        if (
            len(ev.a["onset/global_tick"].intersection(ticks)) > 0
            and len(ev.a["pitch"].intersection(valid_pitches)) > 0
        )
        else ev
        for ev in e
    ]

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
        current_pitches.update(ev.a["pitch"])

    # for current active pitches. replace pitch
    for event_idx in range(n_events):
        if e[event_idx].is_active():
            e[event_idx].a["pitch"] = current_pitches
            e[event_idx] = e[event_idx].force_active()

    return e


def replace_pitches(
    e,
    ec,
    n_events,
    tick_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # set pitch set to all pitches
    for event_idx in range(n_events):
        if e[event_idx].is_active():
            e[event_idx].a["pitch"] = ec().a["pitch"]
            e[event_idx] = e[event_idx].force_active()

    return e

def replace_onset_offsets(
    e,
    ec,
    n_events,
    tick_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # set pitch set to all pitches
    for event_idx in range(n_events):
        if e[event_idx].is_active():
            e[event_idx].a["onset/global_tick"] = ec().a["onset/global_tick"]
            e[event_idx].a["offset/global_tick"] = ec().a["offset/global_tick"]
            e[event_idx] = e[event_idx].force_active()

    return e


def hihat_beat(
    e,
    ec,
    n_events,
    tick_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    e = []
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # remove all drums
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]

    # add 10 kicks
    e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(20)]

    # add 4 snares
    e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(4)]

    # add 10 hihats
    e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(40)]

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

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    # set to 96
    e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

    # set tag to funk
    e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

    return e

from constraints.core import TOM_PITCHES, HIHAT_PITCHES
# create breakbeat


def ride_tom_beat(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    e = []
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # remove drums
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
    # add 10 kicks
    e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(20)]
    # add 10 optional kicks
    e += [ec().intersect({"pitch": {"36 (Drums)", "-"}}) for _ in range(10)]
    # add 3 toms
    e += [ec().intersect({"pitch": TOM_PITCHES}).force_active() for _ in range(10)]

    # add ride
    RIDE_PITCHES = {"51 (Drums)", "53 (Drums)", "59 (Drums)"}
    # add 20 rides
    e += [ec().intersect({"pitch": RIDE_PITCHES}).force_active() for _ in range(40)]
    # 20 optional rides
    e += [ec().intersect({"pitch": RIDE_PITCHES | {"-"}}) for _ in range(20)]
    # add 10 snare
    e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(6)]
    # add 10 optional snares
    e += [ec().intersect({"pitch": {"38 (Drums)", "-"}}) for _ in range(10)]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

    # set tempo
    e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

    # tempo

    return e


def replace_onsets_offsets_given_onset_offset_set(
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
    current_onsets = set()
    current_offsets = set()
    for ev in e:
        current_onsets.update(ev.a["onset/global_tick"])
        current_offsets.update(ev.a["offset/global_tick"])

    # for current active pitches. replace pitch
    for event_idx in range(n_events):
        if e[event_idx].is_active():
            e[event_idx].a["onset/global_tick"] = current_onsets
            e[event_idx].a["offset/global_tick"] = current_offsets
            e[event_idx] = e[event_idx].force_active()

    return e

def unconditional(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
    return [ec() for _ in range(n_events)]

# Define tasks with their parameters
TASKS = {
    "constrained_generation": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": constrained_generation,
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": None,
        "tempo": None,
    },
    "replace_bass_notes": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": replace_bass_notes,
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": None,
        "tempo": None,
    },
    "replace_pitches": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": replace_pitches,
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": None,
        "tempo": None,
    },
    "replace_pitches_given_pitch_set": {
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
    "unconditional": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": unconditional,
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
        "fn": replace_notes_in_box,
        "tick_range": (4*24, 12*24),
        "pitch_range": (22, 110),
        "drums": False,
        "tag": None,
        "tempo": None,
    },
    "hihat_beat":{
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": hihat_beat,
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": "pop",
        "tempo": 96,
    },
    "ride_tom_beat":{
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": ride_tom_beat,
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": "pop",
        "tempo": 160,
    },
    "piano_in_scale_pitch_set":{
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": piano_in_scale_pitch_set,
        "tick_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": "pop",
        "tempo": 120,
    },
}


def setup_model(checkpoint_path):
    """Load and set up the model."""
    print(f"Loading model from {checkpoint_path}...")
    model = TrainingWrapper.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
    model.eval()
    return model

def load_test_dataset(tokenizer):
    """Load the test dataset."""
    print("Loading test dataset...")

    # Filter function for 4-bar loops
    mmd_4bar_filter_fn = lambda x: "n_bars=4" in x

    # Filter for valid piano tracks
    sm_filter_fn = lambda sm: not any(
        track.program == 0 and not track.is_drum and "piano" not in track.name.lower()
        for track in sm.tracks
    )

    test_ds = MidiDataset(
        cache_path="./data/mmd_loops/tst_midi_records_unique_pr.pt",
        path_filter_fn=mmd_4bar_filter_fn,
        genre_list=tokenizer.config["tags"],
        tokenizer=tokenizer,
        min_notes=16,  # 4 notes per bar minimum
        max_notes=tokenizer.config["max_notes"],
        use_random_shift=False,
        sm_filter_fn=sm_filter_fn,
    )

    return test_ds

records = []

def main():
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)

    # First, save some ground truth examples
    print("\nSaving ground truth examples...")
    ground_truth_dir = OUTPUT_DIR / "ground_truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    # We'll use the first checkpoint's tokenizer to load the dataset
    first_model = setup_model(next(iter(CHECKPOINTS.values())))
    test_dataset = load_test_dataset(first_model.tokenizer)

    # Save N_EXAMPLES ground truth loops
    ground_truth_indices = random.sample(range(len(test_dataset)), N_EXAMPLES)
    ground_truth_examples = [test_dataset[idx] for idx in ground_truth_indices]

    print(ground_truth_examples)
    print(ground_truth_examples[0])

    for i, example in enumerate(ground_truth_examples):
        example_sm = first_model.tokenizer.decode(ground_truth_examples[i]["token_ids"])
        sm_set_track_order(example_sm).dump_midi(ground_truth_dir / f"example_{i}.mid")

    # Process each checkpoint
    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
        print(f"\nProcessing checkpoint: {checkpoint_name}")
        model = setup_model(checkpoint_path)

        checkpoint_dir = OUTPUT_DIR / checkpoint_name

        # Process each task
        for task_name, task_config in TASKS.items():
            print(f"\nProcessing task: {task_name}")

            task_dir = checkpoint_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            # Generate examples using the same indices as ground truth
            for i, idx in enumerate(ground_truth_indices):
                print(f"Generating example {i+1}/{N_EXAMPLES}")

                # Get example from test set
                test_example = ground_truth_examples[i]["token_ids"]

                # convert to midi and save
                test_sm = first_model.tokenizer.decode(test_example)

                # warning: read tempo from example instead
                # Convert to events
                test_events = sm_to_events(test_sm, "pop", model.tokenizer)

                print(f"Test events: {len(test_events)}")

                test_mask = model.tokenizer.event_constraints_to_mask(test_events)

                tic = time.time()
                # apply task function
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

                print(f"Task events: {len(task_events)}")

                # convert to mask
                task_mask = model.tokenizer.event_constraints_to_mask(task_events)

                # get conditional likelihood
                with torch.no_grad():
                    model.eval()
                    log_probs = model.model.conditional_log_likelihood(
                        test_mask.to(DEVICE), task_mask.to(DEVICE)
                    )
                    print(log_probs)
                records.append(
                    {
                        "model": checkpoint_name,
                        "task": task_name,
                        "log_probs": log_probs.item(),
                        "example": i,
                    }
                )

                if GENERATE:
                    with torch.no_grad():
                        # Generate from mask
                        output_mask = model.generate(
                            task_mask,
                            **task_config["sampling_settings"],
                        )[0].argmax(dim=-1)

                    # Convert to midi
                    output_sm = model.tokenizer.decode(output_mask)

                    output_sm = sm_fix_overlap_notes(output_sm)
                    # write
                    sm_set_track_order(output_sm).dump_midi(
                        task_dir / f"generated_{i}.mid"
                    )

    print(records)
    # save records as csv
    import pandas as pd

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "records.csv")


if __name__ == "__main__":
    main()

# %%
