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
from PAPER_CHECKPOINTS import CHECKPOINTS
from util import sm_set_track_order, sm_fix_overlap_notes
from tqdm import tqdm
import os
import symusic
from joblib import Parallel, delayed

# Number of examples to generate per task
N_EXAMPLES = 1000
GENERATE = True
# List of sampling orders to iterate over
SAMPLING_ORDERS = ["left_to_right", "left_to_right_reverse", "random"]
# List of SET_N_NOTES values to iterate over
SET_N_NOTES_VALUES = [True, False]
RENDER_GROUND_TRUTH = True
DEVICES = [4,5,6,7]

OUTPUT_DIR = Path("./artefacts/constrained_generation_4")

def setup_model(checkpoint_path, device):
    """Load and set up the model."""
    print(f"Loading model from {checkpoint_path}...")
    model = TrainingWrapper.load_from_checkpoint(checkpoint_path, map_location=device, weights_only=False)
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
        n_bars=4,
    )
    return test_ds


def main():
    SEED = 0
    # Set random seeds
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    FILTER_DEVICE = DEVICES[0]

    models = ["slm_mixed_150epochs", "slm_sparse_150epochs", "slm_full_150epochs", "mlm_150epochs"]

    TEMPERATURE = 1.0
    COLLAPSE_DUPLICATES = True

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the test dataset using first available model's tokenizer
    # Load the specified model
    first_model = setup_model(CHECKPOINTS[models[0]], "cpu")
  
    # Create and save ground truth examples
    ground_truth_dir = OUTPUT_DIR / "ground_truth"
    reference_dir = ground_truth_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = first_model.tokenizer

    ec = lambda: MusicalEventConstraint(first_model.tokenizer)
    N_EVENTS = tokenizer.config["max_notes"]
    n_beats = 16
    tpq = tokenizer.config["ticks_per_beat"]

    constraints = {
        "piano" : [ ec().intersect({"instrument": {"Piano", "-"}}) for _ in range(N_EVENTS) ],
        "drum_and_bass_and_guitar" : [ ec().intersect({"instrument": {"Drums", "Bass", "Guitar","-"}}) for _ in range(N_EVENTS) ],
        "drum_and_bass_and_piano" : [ ec().intersect({"instrument": {"Drums", "Bass", "Piano","-"}}) for _ in range(N_EVENTS) ],
        "drum_and_bass" : [ ec().intersect({"instrument": {"Drums", "Bass","-"}}) for _ in range(N_EVENTS) ],
        "guitar" : [ ec().intersect({"instrument": {"Guitar","-"}}) for _ in range(N_EVENTS) ],
        "drums" : [ ec().intersect({"instrument": {"Drums","-"}}) for _ in range(N_EVENTS) ],
        "1 d 2 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, 4 * tpq//2)} | {"-"} }) for _ in range(N_EVENTS) ],
        "1 d 4 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, 4 * tpq//4)} | {"-"} }) for _ in range(N_EVENTS) ],
        "1 d 8 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, 4 * tpq//8)} | {"-"} }) for _ in range(N_EVENTS) ],
        "1 d 16 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, 4 * tpq//16)} | {"-"} }) for _ in range(N_EVENTS) ],
        "c major pitch set" : [ ec().intersect({"pitch": ec().pitch_in_scale_constraint("C major", (10, 127))["pitch"] | {"-"} }) for _ in range(N_EVENTS) ],
        "c pentatonic" : [ ec().intersect({"pitch": ec().pitch_in_scale_constraint("C pentatonic", (10, 127))["pitch"] | {"-"} }) for _ in range(N_EVENTS) ],
    }

    if RENDER_GROUND_TRUTH:
        test_dataset = load_test_dataset(first_model.tokenizer)

        filter_batch_size = 100

        dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=filter_batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
        )

        constraint_records = {c:[] for c in constraints.keys()}

        for batch in tqdm(dl):
            token_ids = batch["token_ids"].to(FILTER_DEVICE)
            one_hot = torch.nn.functional.one_hot(token_ids, len(tokenizer.vocab))

            for constraint_name, e in constraints.items():
                mask = tokenizer.event_constraints_to_mask(e).to(FILTER_DEVICE)
                follows_constraint = (one_hot * mask).sum(dim=-1) > 0
                follows_constraint = follows_constraint.all(dim=[1, 2])
                print(token_ids[follows_constraint].shape)
                # get examples that follow the constraint and add to records
                if len(token_ids[follows_constraint]) > 0:
                    constraint_records[constraint_name].extend(token_ids[follows_constraint].cpu().split(1))

        # show statistics
        for constraint_name, examples in constraint_records.items():
            print(f"Found {len(examples)} examples for constraint {constraint_name}")

        # save found examples as midi
        for constraint_name, examples in constraint_records.items():
            os.makedirs(ground_truth_dir / constraint_name, exist_ok=True)
            print(f"Rendering {constraint_name}")
            for i, example in tqdm(enumerate(examples)):
                # remove 0 dimension on first axis
                example_sm = first_model.tokenizer.decode(example[0])
                sm_set_track_order(example_sm).dump_midi(ground_truth_dir / f"{constraint_name}/{i}.mid")

        # for each constraint, load midi
        gt_records = []

        for constraint_name, e in constraints.items():
            ground_truth_midi_dir = ground_truth_dir / constraint_name
            midi_files = list(ground_truth_midi_dir.glob("*.mid"))
            for i, midi_file in tqdm(enumerate(midi_files)):
                # load midi file
                sm = symusic.Score(midi_file)
                n_notes = sm.note_num()
                # add record
                gt_records.append({
                    "constraint": constraint_name,
                    "midi_file_path": str(midi_file),
                    "n_notes": n_notes,
                })
                
        # compute for each constraint the number of examples, min max mean and median for n_notes
        gt_records_df = pd.DataFrame(gt_records)
        gt_records_df.to_csv(ground_truth_dir / "ground_truth_records.csv", index=False)

    # load records
    gt_records_df = pd.read_csv(ground_truth_dir / "ground_truth_records.csv")

    # now render examples with each model, sampling order, and SET_N_NOTES setting
    def render_examples_with_model(model_name, constraints, output_dir, n_examples, device, order, set_n_notes):
        # for each constraint, generate examples and save under model/constraint_name directory
        model = setup_model(CHECKPOINTS[model_name], device)
        system_name = f"{model_name}_order={order}_t={TEMPERATURE}_set_n_notes={set_n_notes}"
        for constraint_name, e in constraints.items():
            task_dir = output_dir / system_name / constraint_name
            task_dir.mkdir(parents=True, exist_ok=True)
            # get list of ground_truth example midi files
            n_note_per_example = gt_records_df[gt_records_df["constraint"] == constraint_name]["n_notes"].tolist()
            generation_records = []
            for i in tqdm(range(n_examples)):
                try:
                    # get n_notes from gt records for constraint
                    if set_n_notes and i < len(n_note_per_example):
                        n_events = 300
                        n_notes = n_note_per_example[i]
                        e_bis = [e[0].copy().force_active() for _ in range(n_notes)]
                        e_bis += [e[1].copy().force_inactive() for _ in range(n_events - len(e_bis))]
                    else:
                        e_bis = e
                    print(f"Generating example {i+1}/{n_examples} for {model_name}, order={order}, set_n_notes={set_n_notes}, constraint={constraint_name}")
                    mask = model.tokenizer.event_constraints_to_mask(e_bis).to(device)
                    x = model.generate(
                        mask,
                        temperature=TEMPERATURE,
                        top_p=1.0,
                        top_k=0,
                        tokens_per_step=1,
                        order=order,
                        collapse_duplicates=COLLAPSE_DUPLICATES,
                    )[0].argmax(-1)
                    x_sm = model.tokenizer.decode(x)
                    sm_set_track_order(x_sm).dump_midi(task_dir / f"generated_{i}.mid")
                    generation_records.append({
                        "model": model_name,
                        "system_name": system_name,
                        "constraint": constraint_name,
                        "example_index": i,
                        "order": order,
                        "set_n_notes": set_n_notes
                    })
                except Exception as error:
                    print(f"Error: {error}")
                    print(f"Generating example {i+1}/{n_examples} failed for {model_name} with constraint {constraint_name}")
            # write generation records
            df = pd.DataFrame(generation_records)
            df.to_csv(task_dir / "generation_records.csv", index=False)
            
    # Generate examples for each model, sampling order, and SET_N_NOTES setting
    if GENERATE:
        # Create arguments for each job
        args = []
        job_idx = 0
        for model in models:
            for order in SAMPLING_ORDERS:
                for set_n_notes in SET_N_NOTES_VALUES:
                    args.append((model, constraints, OUTPUT_DIR, N_EXAMPLES, 
                                f"cuda:{DEVICES[job_idx % len(DEVICES)]}", order, set_n_notes))
                    job_idx += 1

        # Run jobs in parallel with a limited number of jobs at once to prevent GPU memory issues
        n_parallel_jobs = len(DEVICES)  # Adjust based on available GPU memory
        Parallel(n_jobs=n_parallel_jobs, prefer="threads")(
            delayed(render_examples_with_model)(model, constraints, OUTPUT_DIR, N_EXAMPLES, device, order, set_n_notes)
            for model, constraints, OUTPUT_DIR, N_EXAMPLES, device, order, set_n_notes in args
        )

        print("All model generation complete!")

        # Create a summary DataFrame of all configurations
        all_records = []
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                if file == "generation_records.csv":
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        all_records.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        if all_records:
            combined_df = pd.concat(all_records, ignore_index=True)
            combined_df.to_csv(OUTPUT_DIR / "all_generation_records.csv", index=False)
            print(f"Summary report saved to {OUTPUT_DIR / 'all_generation_records.csv'}")


if __name__ == "__main__":
    main()