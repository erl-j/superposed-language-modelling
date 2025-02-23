#%%
from datasets import load_dataset
from tqdm import tqdm
import symusic
from util import preview_sm
import pandas as pd
import torch
#%%
# import torch
# # load pr
# example = "../data/gmd_loops/val_midi_records.pt"
# template_ds = torch.load(example)
# print(template_ds[0][0].keys())
# print(template_ds[0][0])

#%%
TPQ = 96
LIMIT = 1_000_000_000

ds_path = "./data/gmd_loops/"

def preprocess_sample(sample):
    # remove the midi file from the sample
    score = symusic.Score.from_midi(sample["music"])
    # 
    # get resolution
    original_tpq = score.tpq
    score = score.resample(TPQ, min_dur=0)

    qpm = score.tempos[-1].qpm
    time_sig = f"{score.time_signatures[-1].numerator}/{score.time_signatures[-1].denominator}"

    loops = sample["loops"]
    loop_records = []
    for i in range(len(loops["track_idx"])):
        loop_record = {key: loops[key][i] for key in loops.keys()}
        loop_records.append(loop_record)   

    # next, group loops that share the same start and end ticks such that we have a record:
    # {"start_tick": start_tick, "end_tick": end_tick, "track_idxs": [track_idxs]}
    loop_records = sorted(loop_records, key=lambda x: (x["start_tick"], x["end_tick"]))
    loop_records_grouped = []
    for loop in loop_records:
        if not loop_records_grouped or loop_records_grouped[-1]["start_tick"] != loop["start_tick"] or loop_records_grouped[-1]["end_tick"] != loop["end_tick"]:
            loop_records_grouped.append({"start_tick": loop["start_tick"], "end_tick": loop["end_tick"], "track_idxs": []})
        loop_records_grouped[-1]["track_idxs"].append(loop["track_idx"])

    # resample loop ticks to new tpq
    for loop in loop_records_grouped:
        loop["start_tick"] = loop["start_tick"] * TPQ // original_tpq
        loop["end_tick"] = loop["end_tick"] * TPQ // original_tpq

    return {**sample, "midi": score, "loops": loop_records_grouped, "qpm": qpm, "time_sig": time_sig}

genre_set = open(f"{ds_path}/tags.txt").read().splitlines()

split_shorthands = {
    "train": "trn",
    "validation": "val",
    "test": "tst"
}

def crop_sm(sm, n_bars, beats_per_bar):
    sm = sm.copy()
    end_tick = n_bars * sm.ticks_per_quarter * beats_per_bar
    for track_idx in range(len(sm.tracks)):
        track = sm.tracks[track_idx]
        new_notes = []
        for note in track.notes:
            if note.start < end_tick:
                # crop duration
                if note.start + note.duration > end_tick:
                    note.duration = end_tick - note.start
                new_notes.append(note)
        sm.tracks[track_idx].notes = new_notes
    return sm


def get_loop_records(record):
    loop_records = []
    genres = record["genres_scraped"] if record["genres_scraped"] is not None else []
    # keep only genres in genre_list
    genres = [genre for genre in genres if genre in genre_set]
    # make sure genres are unique
    genres = list(set(genres))
    for loop in record["loops"]:
        loop_sm = record["midi"].clip(
        start=loop["start_tick"],
        end=loop["end_tick"],
        clip_end=True
        ).shift_time(-loop["start_tick"])

        loop_time = loop["end_tick"] - loop["start_tick"]
        # remove notes that start after the loop ends
                   
        loop_bars = (loop["end_tick"] - loop["start_tick"]) // (TPQ * 4)

        loop_sm = crop_sm(loop_sm, loop_bars, 4)

        loop_records.append(
            {
                "path" : f"gigamidi_n_bars={loop_bars}",
                "md5": record["md5"],
                "genre": genres,
                "midi": loop_sm,
            }
        )   
    return loop_records


for split in ["test","validation","train"]:
    dataset = load_dataset("Metacreation/GigaMIDI", split=split)
    records = []
    sample_idx = 0
    for sample in tqdm(dataset):
        try:
            sample_idx += 1
            records.append(preprocess_sample(sample))
        except Exception as e:
            print(e)
        if sample_idx >= LIMIT:
            break
    print(f"Number of records: {len(records)}")

    # filter 4/4
    records = [record for record in records if record["time_sig"] == "4/4"]

    # get loop records
    loop_records = []
    for record in records:
        loop_records.append(get_loop_records(record))

    # save 
    torch.save(loop_records, f"{ds_path}/{split_shorthands[split]}_midi_records.pt")