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

data_path = "../data/harmonic"
NUM_BARS = 8

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
    midi_sm = midi_sm.resample(tpq=TPQ,min_dur=1)
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


def loop_sm(sm, loop_bars, beats_per_bar):
    sm = sm.copy()
    loop_ticks = loop_bars * sm.ticks_per_quarter * beats_per_bar
    sm = crop_sm(sm, loop_bars, beats_per_bar)
    for track_idx in range(len(sm.tracks)):
        new_notes = []
        for note in sm.tracks[track_idx].notes:
            old_note = symusic.Note(
                time=note.start,
                pitch=note.pitch,
                duration=note.duration,
                velocity=note.velocity,
            )
            new_note = symusic.Note(
                time=note.start + loop_ticks,
                pitch=note.pitch,
                duration=note.duration,
                velocity=note.velocity,
            )
            new_notes.append(old_note)
            new_notes.append(new_note)
        sm.tracks[track_idx].notes = new_notes
    return sm


def get_last_onset(sm):
    sm = sm.copy()
    last_onset = 0
    for track in sm.tracks:
        for note in track.notes:
            last_onset = max(last_onset, note.start)
    return last_onset


def get_n_bars(sm, beats_per_bar=4):
    sm=sm.copy()
    last_onset = get_last_onset(sm)
    n_beats = last_onset / sm.ticks_per_quarter
    # find closest multiple of 2
    n_bars = math.ceil(n_beats / beats_per_bar)
    return n_bars


def loop_crop_to(sm, n_bars, beats_per_bar):
    sm = sm.copy()
    old_bars = get_n_bars(sm, beats_per_bar)
    looped_sm = sm
    while old_bars <= n_bars:
        looped_sm = loop_sm(looped_sm, old_bars, beats_per_bar)
        new_bars = get_n_bars(looped_sm, beats_per_bar)
        if new_bars == old_bars:
            break
        else:
            old_bars = new_bars
    looped_sm = crop_sm(looped_sm, 4, beats_per_bar)
    return looped_sm


def remove_invalid_notes(sm):
    sm = sm.copy()
    for track_idx in range(len(sm.tracks)):
        new_notes = []
        for note in sm.tracks[track_idx].notes:
            if note.start >= 0 or note.duration >= 0:
                new_notes.append(note)
            else:
                print(f"Invalid note: {note}")
        sm.tracks[track_idx].notes=new_notes
    return sm


for i in tqdm(range(len(records))):
    records[i]["midi"] = loop_crop_to(records[i]["raw_midi"], NUM_BARS, beats_per_bar=4)
    records[i]["midi"] = remove_invalid_notes(records[i]["midi"])

# preview midi for 10 random records

# from util import preview_sm
# record = records[2]
# raw_sm = record["raw_midi"]
# # print n_beats
# # get last onset
# print(f"last_onset: {get_last_onset(raw_sm)}")
# print(f"n_bars {get_n_bars(raw_sm)}")

# # crop to 4 beats


# looped_sm = loop_crop_to(raw_sm, 2, beats_per_bar=4)

# print(f"last_onset: {get_last_onset(looped_sm)}")
# print(f"n_bars {get_n_bars(looped_sm)}")

# preview_sm(raw_sm)
# preview_sm(looped_sm)
# crop 2 beats
# croped_sm = crop_sm(raw_sm, 2, beats_per_bar=4)
# print(f"n_bars {get_n_bars(croped_sm)}")
# print(f"last_onset: {get_last_onset(croped_sm)}")


# # try loop crop to 16 beats
# new_sm = loop_crop_to(raw_sm, 16)
# print(get_n_beats(new_sm))
# preview_sm(raw_sm)
# preview_sm(new_sm)

#%%

#%%

#%%



#%%
import matplotlib.pyplot as plt
records = [{"note_num": record["raw_midi"].note_num(), **record} for record in records]

# plot histogram of note_num
note_nums = [record["note_num"] for record in records]
plt.hist(note_nums, bins=100, range=(0, 30))
plt.show()

# plot note num vs ticks
note_nums = [record["note_num"] for record in records]
beats = [get_last_onset(record["raw_midi"])
         /record["raw_midi"].ticks_per_quarter for record in records]

plt.scatter(note_nums, beats)
# set y lim from 0 to 20
plt.ylim(0, 32)
plt.show()

# plot histogram of num ticks
beats = [get_last_onset(record["raw_midi"])
         /record["midi"].ticks_per_quarter for record in records]
plt.hist(beats, bins=100)
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
