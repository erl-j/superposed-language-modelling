import os
import torch
import random
from pathlib import Path
import symusic
from train import TrainingWrapper
from data import MidiDataset
from conversion_utils import looprep_to_sm, sm_to_events, sm_to_looprep
from constraints.core import MusicalEventConstraint

# Configuration
SEED = 42
DEVICE = "cuda:6" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("./artefacts/applications")

# Model checkpoints to test
CHECKPOINTS = {
    "slm": "./checkpoints/usual-fire-530/last.ckpt",
    "mlm": "./checkpoints/toasty-bush-529/last.ckpt",
}

# Number of examples to generate per task
N_EXAMPLES = 5


def replace_bass_notes(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
    ):
        # count n bass notes
        n_bass_notes = len([ev for ev in e if ev.a["instrument"] == {"Bass"}])
        # remove all bass
        e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]

        # add back bass notes
        e += [
            ec()
            .intersect({"instrument": {"Bass"}})
            .force_active()
            for _ in range(n_bass_notes)
        ]

        # pad with empty notes
        e += [ec().force_inactive() for _ in range(n_events - len(e))]

        # # add tag constraint
        # e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

        # # add tempo constraint
        # e = [ev.intersect({"tempo": {tempo, "-"}}) for ev in e]

        return e

# Define tasks with their parameters
TASKS = {
    "replace_bass_notes": {
        "sampling_settings": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "tokens_per_step": 1,
        },
        "fn": replace_bass_notes,
        "beat_range": None,
        "pitch_range": None,
        "drums": None,
        "tag": None,
        "tempo": None,
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

    for i, example in enumerate(ground_truth_examples):
        example_sm =  first_model.tokenizer.decode(ground_truth_examples[i]["token_ids"])
        example_sm.dump_midi(ground_truth_dir / f"example_{i}.mid")
            
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


                # apply task function
                task_events = task_config["fn"](
                    test_events,
                    lambda : MusicalEventConstraint(model.tokenizer),
                    model.tokenizer.config["max_notes"],
                    task_config["beat_range"],
                    task_config["pitch_range"],
                    task_config["drums"],
                    task_config["tag"],
                    task_config["tempo"],
                )

                # convert to mask
                task_mask = model.tokenizer.event_constraints_to_mask(task_events)

                # Generate from mask
                output_mask = model.generate(
                    task_mask,
                    **task_config["sampling_settings"],
                )[0].argmax(dim=-1)

                # Convert to midi
                output_sm = model.tokenizer.decode(output_mask)

                # write
                output_sm.dump_midi(task_dir / f"generated_{i}.mid")

if __name__ == "__main__":
    main()
