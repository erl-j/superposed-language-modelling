import os
import torch
import random
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import time
from slm.train import TrainingWrapper
from data import MidiDataset
from conversion_utils import sm_to_events
from constraints.core import MusicalEventConstraint
from slm.PAPER_CHECKPOINTS import CHECKPOINTS
from util import sm_set_track_order, sm_fix_overlap_notes
from tqdm import tqdm
import os

# Number of examples to generate per task
N_EXAMPLES = 25
GENERATE = True
ORDER = "random"

OUTPUT_DIR = Path("./artefacts/constrained_generation")

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
        n_bars=4,
    )
    return test_ds


def main():
    SEED = 0
    # Set random seeds
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    models = ["slm_mixed_150epochs", "slm_sparse_150epochs", "slm_full_150epochs", "mlm_150epochs"]

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    # Load the test dataset using first available model's tokenizer
    # Load the specified model
    first_model = setup_model(models[0], "cpu")

    test_dataset = load_test_dataset(first_model.tokenizer)
  
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
        "drum_and_bass" : [ ec().intersect({"instrument": {"Drums", "Bass","-"}}) for _ in range(N_EVENTS) ],
        "1 d 2 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, 4 * tpq//2)} | {"-"} }) for _ in range(N_EVENTS) ],
        "1 d 4 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, 4 * tpq//4)} | {"-"} }) for _ in range(N_EVENTS) ],
        "1 d 8 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, 4 * tpq//8)} | {"-"} }) for _ in range(N_EVENTS) ],
        "1 d 16 notes" : [ ec().intersect({"onset/global_tick": {str(t) for t in range(0, n_beats*tpq, 4 * tpq//16)} | {"-"} }) for _ in range(N_EVENTS) ],
        "c major pitch set" : [ ec().intersect({"pitch": ec().pitch_in_scale_constraint("C major", (10, 127))["pitch"] | {"-"} }) for _ in range(N_EVENTS) ],
        "c pentatonic" : [ ec().intersect({"pitch": ec().pitch_in_scale_constraint("C pentatonic", (10, 127))["pitch"] | {"-"} }) for _ in range(N_EVENTS) ],
    }

    filter_batch_size = 100

    dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=filter_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    constraint_records = {c:[] for c in constraints.keys()}

    for batch in tqdm(dl):
        token_ids = batch["token_ids"]
        one_hot = torch.nn.functional.one_hot(token_ids, len(tokenizer.vocab))

        for constraint_name, e in constraints.items():
            mask = tokenizer.event_constraints_to_mask(e)
            follows_constraint = (one_hot * mask).sum(dim=-1) > 0
            follows_constraint = follows_constraint.all(dim=[1, 2])
            # get examples that follow the constraint and add to records
            constraint_records[constraint_name].extend(token_ids[follows_constraint].split(0))

    # show statistics
    for constraint_name, examples in constraint_records.items():
        print(f"Found {len(examples)} examples for constraint {constraint_name}")


            


    # # constraint_records = {}
    # # for constraint_name, e in constraints.items():
    # #     mask = tokenizer.event_constraints_to_mask(e)
    # #     # get test data examples that respect the constraint
    # #     # filter data by constraint.
    # #     natural_examples = []
    # #     n_found_examples = 0
    # #     for batch in dl:
    # #         token_ids = batch["token_ids"]
    # #         one_hot = torch.nn.functional.one_hot(token_ids, len(model.tokenizer.vocab))
    # #         follows_constraint = (one_hot * mask).sum(dim=-1) > 0
    # #         follows_constraint = follows_constraint.all(dim=[1, 2])

    # #         # get examples that follow the constraint
    # #         # pick examples that follow the constraint
    # #         for i in range(len(follows_constraint)):
    # #             if follows_constraint[i]:
    # #                 natural_examples.append(token_ids[i])
    # #                 n_found_examples += 1
    # #     # save how many examples we found
    # #     constraint_records[constraint_name] = {
    # #         "constraint": constraint_name,
    # #         "n_found_examples": n_found_examples,
    # #         "natural_examples": natural_examples[:N_EXAMPLES],
    # #     }

    #     # save found examples in ground truth dir as text file
    #     # # Save records for this model
    #     # df = pd.DataFrame(constraint_records)
    #     # df.to_csv(f"{str(ground_truth_dir)}/constraint_records.csv", index=False)

    # constraint_to_n_notes = {}
    # for constraint_name, e in constraints.items():
    #     mask = model.tokenizer.event_constraints_to_mask(e)

    #     n_found_examples = constraint_records[constraint_name]["n_found_examples"]
    
    #     # count how many examples we found
    #     print(f"Found {n_found_examples} examples for constraint {constraint_name}")

    #     # crop to N_EXAMPLES
    #     natural_examples = constraint_records[constraint_name]["natural_examples"]

    #     # save n found examples in ground truth dir as text file
    #     if args.render_ground_truth:
    #         with open(ground_truth_dir / f"{constraint_name}_n_found_examples.txt", "w") as f:
    #             f.write(f"{n_found_examples}\n")


    #     os.makedirs(ground_truth_dir / constraint_name, exist_ok=True)
    #     # render examples
    #     constraint_to_n_notes[constraint_name] = []
    #     for i, example in enumerate(natural_examples):
    #         example_sm = first_model.tokenizer.decode(example)
    #         constraint_to_n_notes[constraint_name].append(example_sm.note_num())

    #         if args.render_ground_truth:
    #             sm_set_track_order(example_sm).dump_midi(ground_truth_dir / f"{constraint_name}/{i}.mid")

    #     if False:
    #         # now generate examples
    #         task_dir = Path(checkpoint_dir) / constraint_name
    #         task_dir.mkdir(parents=True, exist_ok=True)

    #         records = []

    #         min_notes = min(constraint_to_n_notes[constraint_name])
    #         max_notes = max(constraint_to_n_notes[constraint_name])

    #         e_bis = []
    #         e_bis += [e[0].copy().force_active() for i in range(min_notes)]
    #         e_bis += [e[0].copy() for i in range(max_notes - min_notes)]
    #         e_bis += [e[0].copy().force_inactive() for i in range(N_EVENTS - max_notes)]
    #         e = e_bis
    #         for i in range(N_EXAMPLES):
            
    #             print(f"Generating example {i+1}/{N_EXAMPLES}")
    #             mask = model.tokenizer.event_constraints_to_mask(e).to(device)
    #             x = model.generate(
    #                 mask,
    #                 temperature=1.0,
    #                 top_p=1.0,
    #                 top_k=0,
    #                 tokens_per_step=1,
    #                 order=ORDER,
    #             )[0].argmax(-1)
    #             x_sm = model.tokenizer.decode(x)
    #             x_sm = sm_fix_overlap_notes(x_sm)
    #             sm_set_track_order(x_sm).dump_midi(task_dir / f"generated_{i}.mid")

    #             records.append({
    #                 "model": args.model,
    #                 "task": constraint_name,
    #                 "example": i,
    #             })

    #         # Save records for this model
    #         df = pd.DataFrame(records)
    #         df.to_csv(f"{str(task_dir)}/records.csv", index=False)
    # # Process each task
    # for task_name, task_config in TASKS.items():
    #     records = []
    #     print(f"\nProcessing task: {task_name}")
    #     task_dir = Path(checkpoint_dir) / task_name
    #     task_dir.mkdir(parents=True, exist_ok=True)

    #     for i, idx in enumerate(ground_truth_indices):
    #         print(f"Generating example {i+1}/{N_EXAMPLES}")
            
    #         # Get example from test set
    #         test_example = ground_truth_examples[i]["token_ids"]
    #         test_sm = first_model.tokenizer.decode(test_example)
    #         test_events = sm_to_events(test_sm, "pop", model.tokenizer)
            
    #         # Apply task function
    #         tic = time.time()
    #         task_events = task_config["fn"](
    #             test_events,
    #             lambda: MusicalEventConstraint(model.tokenizer),
    #             model.tokenizer.config["max_notes"],
    #             task_config["tick_range"],
    #             task_config["pitch_range"],
    #             task_config["drums"],
    #             task_config["tag"],
    #             task_config["tempo"],
    #         )
    #         toc = time.time()
    #         print(f"Time taken to apply task function: {toc-tic}")

    #         # Convert to mask and get conditional likelihood
    #         task_mask = model.tokenizer.event_constraints_to_mask(task_events)
    #         with torch.no_grad():
    #             model.eval()
    #             test_mask = model.tokenizer.event_constraints_to_mask(test_events)
    #             log_probs = model.model.conditional_log_likelihood(
    #                 test_mask.to(device), task_mask.to(device)
    #             )
    #             print(f"Log probs: {log_probs.item()}")
                
    #             records.append({
    #                 "model": args.model,
    #                 "task": task_name,
    #                 "log_probs": log_probs.item(),
    #                 "example": i,
    #             })
                
    #             if GENERATE:
    #                 output_mask = model.generate(
    #                     task_mask,
    #                     **task_config["sampling_settings"],
    #                     order=ORDER,
    #                 )[0].argmax(dim=-1)
                    
    #                 try:
    #                     output_sm = model.tokenizer.decode(output_mask)
    #                     output_sm = sm_fix_overlap_notes(output_sm)
    #                     sm_set_track_order(output_sm).dump_midi(task_dir / f"generated_{i}.mid")
    #                 except Exception as e:
    #                     print(f"Error decoding MIDI: {e}")

    #     # Save records for this model
    #     df = pd.DataFrame(records)
    #     df.to_csv(f"{str(task_dir)}/records.csv", index=False)

if __name__ == "__main__":
    main()