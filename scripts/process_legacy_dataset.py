#%%
import json
import os
import pickle
import note_seq
from rich.progress import track
import symusic
import torch
#%%

# example_records = torch.load("./data/mmd_loops/val_midi_records_unique_pr.pt")
# print(len(example_records))
# print(example_records[0][0].keys())
# path, md5, genre, midi

#%%

data_path = "./data/clean_drums"

# load pickle
pickle_path = os.path.join(data_path, "clean.pkl")

with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

tmp_midi_path = os.path.join(data_path, "tmp_midi_files")
os.makedirs(tmp_midi_path, exist_ok=True)

metadata_path = os.path.join(data_path, "meta.json")
meta = json.loads(open(metadata_path).read())

#%%

records = []
#'path', 'md5', 'genre', 'midi'
for i in range(len(data)):
    tmp_midi_fp = os.path.join(tmp_midi_path, f"{i}.mid")
    # export midi
    note_seq.midi_io.note_sequence_to_midi_file(data[i]["note_seq"], tmp_midi_fp)
    # load midi with symusic
    midi_sm = symusic.Score(tmp_midi_fp)
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
     
    N_BARS = 4
    bars_ticks = tpq * 4 * N_BARS

    ##
    # remove everything after 4 bars
    end_4_bars_tick = 16 * midi_sm.ticks_per_quarter

    print(f"end of midi file: {end_4_bars_tick}")

    # print(f"end of midi file: {end_4_bars_tick}")
    # print(f"end of midi file: {midi_sm.end()}")

    if midi_sm.end() > end_4_bars_tick or midi_sm.end() <= end_4_bars_tick*(3/4):
        continue
    else:
        print(f"{midi_sm.end()} midi file is just right")
        path = data[i]["path"]
        md5 = i
        midi = midi_sm
        tag = meta[path]["tags"]
        records.append({"path": path, "md5": md5, "genre": tag, "midi": midi})
        
# %%

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
