#%%
import json
import os
import pickle
import note_seq
from rich.progress import track
import symusic
import torch
from tqdm import tqdm
# add slm to path
import sys
sys.path.append("../slm")
import math
#%%

# example_records = torch.load("./data/mmd_loops/val_midi_records_unique_pr.pt")
# print(len(example_records))
# print(example_records[0][0].keys())
# path, md5, genre, midi

#%%

data_path = "../data/clean_drums"
NUM_BARS = 4

# load pickle
pickle_path = os.path.join(data_path, "clean.pkl")

with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

tmp_midi_path = os.path.join(data_path, "tmp_midi_files")
os.makedirs(tmp_midi_path, exist_ok=True)

metadata_path = os.path.join(data_path, "meta.json")
meta = json.loads(open(metadata_path).read())

#%%

TPQ = 240
records = []
#'path', 'md5', 'genre', 'midi'
for i in tqdm(range(len(data))):
    tmp_midi_fp = os.path.join(tmp_midi_path, f"{i}.mid")
    # export midi
    note_seq.midi_io.note_sequence_to_midi_file(data[i]["note_seq"], tmp_midi_fp)
    # load midi with symusic
    midi_sm = symusic.Score(tmp_midi_fp)
    midi_sm = midi_sm.resample(tpq=TPQ,min_dur=0)
    original_midi_path = data[i]["path"]
    # path
    # get key signature
    # if time signature is not 4/4, then resample

    # filter out non 4/4 time signatures
    num = midi_sm.time_signatures[-1].numerator
    denum = midi_sm.time_signatures[-1].denominator

    if num != 4 or denum != 4:
        continue
    # crop to 4 first bars
    tpq = midi_sm.ticks_per_quarter
    # print(f"end of midi file: {end_4_bars_tick}")
    # print(f"end of midi file: {midi_sm.end()}")

    path = data[i]["path"]
    md5 = i
    midi = midi_sm
    tag = meta[path]["tags"]
    records.append({"path": path, "md5": md5, "genre": tag, "raw_midi": midi, "ticks": midi_sm.end()})
        
# %%

import matplotlib.pyplot as plt
import numpy as np

ticks = [record["midi"].end() / record["midi"].ticks_per_quarter for record in records]
plt.hist(ticks, bins=100, range=(0, 16 * 4))

#%%

def crop_sm(sm, num_beats=4):
    sm = sm.copy()
    end_4_bars_tick = num_beats * sm.ticks_per_quarter
    for track in sm.tracks:
        new_notes = []
        for note in track.notes:
            if note.start < end_4_bars_tick:
                note.duration = min(end_4_bars_tick - note.start, note.duration)
                new_notes.append(note)
        track.notes = new_notes
    return sm

def loop_sm(sm, loop_beats):
    sm = sm.copy()
    loop_ticks = loop_beats * sm.ticks_per_quarter
    for track in sm.tracks:
        for note in track.notes:
            new_note = symusic.Note(
                time=note.start + loop_ticks,
                pitch=note.pitch,
                duration=note.duration,
                velocity=note.velocity,
            )
            track.notes.append(new_note)
    crop_sm(sm, loop_beats*2)
    return sm

def get_last_onset(sm):
    last_onset = 0
    for track in sm.tracks:
        for note in track.notes:
            last_onset = max(last_onset, note.start)
    return last_onset


def get_n_beats(sm):
    last_onset = get_last_onset(sm)
    last_beat = last_onset / sm.ticks_per_quarter
    # find closest multiple of 4
    n_beats = math.ceil(last_beat / 4)*4
    return n_beats

def loop_crop_to(sm, n_beats):
    sm = sm.copy()
    beats = get_n_beats(sm)
    looped_sm = sm
    while beats <= n_beats:
        looped_sm = loop_sm(looped_sm, beats)
        beats = get_n_beats(looped_sm)
    looped_sm = crop_sm(looped_sm, n_beats)
    return looped_sm

def remove_invalid_notes(sm):
    sm = sm.copy()
    for track in sm.tracks:
        new_notes = []
        for note in track.notes:
            if note.start >= 0 or note.duration >= 0:
                new_notes.append(note)
            else:
                print(f"Invalid note: {note}")
        track.notes = new_notes
    return sm

#%% 
for i in tqdm(range(len(records))):
    records[i]["midi"] = loop_crop_to(records[i]["raw_midi"],16)
    records[i]["midi"] = remove_invalid_notes(records[i]["midi"])

#%%
records = [{"note_num": record["midi"].note_num(), **record} for record in records]

# plot histogram of note_num
note_nums = [record["note_num"] for record in records]
plt.hist(note_nums, bins=100, range=(0, 300))
plt.show()

#%%

# add one list level
records = [[record] for record in records]
print(len(records))
# split into train test val according to 90/10/10

#%%
# shuffle
import random
random.seed(0)
random.shuffle(records)

DEV_RATIO = 0.9
TRN_RATIO = 0.9

n = len(records)
n_dev = int(n * DEV_RATIO)
n_trn = int(n_dev * TRN_RATIO)

dev_records = records[:n_dev]
tst_records = records[n_dev:]
trn_records = dev_records[:n_trn]
val_records = dev_records[n_trn:]

# save to pytorch tensors
torch.save(trn_records, data_path + "/trn_midi_records_unique_pr.pt")
torch.save(val_records, data_path + "/val_midi_records_unique_pr.pt")
torch.save(tst_records, data_path + "/tst_midi_records_unique_pr.pt")

# %%
print(len(trn_records))
# %%
print(len(val_records))
# %%
print(len(tst_records))
# %%
