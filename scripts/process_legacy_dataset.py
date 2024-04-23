#%%
import json
import os
import pickle
import note_seq
from rich.progress import track
import symusic
import torch
#%%


# example_records = torch.load("../data/mmd_loops/val_midi_records_unique_pr.pt")

# path, md5, genre, midi

#%%

data_path = "../data/clean_drums"

# load pickle
pickle_path = os.path.join(data_path, "clean.pkl")

with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

tmp_midi_path = os.path.join(data_path, "tmp_midi_files")
os.makedirs(tmp_midi_path, exist_ok=True)

metadata_path = os.path.join(data_path, "meta.json")
meta = json.loads(open(metadata_path).read())

#%%
print(data[0])

#%%

for i in track(range(len(data))):
    tmp_midi_fp = os.path.join(tmp_midi_path, f"{i}.mid")
    # export midi
    note_seq.midi_io.note_sequence_to_midi_file(data[i]["note_seq"], tmp_midi_fp)
    # load midi with symusic
    midi_sm = symusic.Score(tmp_midi_fp)
    original_midi_path = data[i]["path"]
    # path
    tags = meta[original_midi_path]["tags"]

    # get n bars
    n_bars = len(midi_sm.get_bars())









# %%
