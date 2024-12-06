# %%

import symusic as sm
import torch
import tempfile
import muspy
from tqdm import tqdm

ckpt = "../data/mmd_loops/val_midi_records_unique_pr.pt"

records = torch.load(ckpt)


# flat the records
records = [r for record in records for r in record]

# preserve only those with "n_bars=4.0" in the path
records = [record for record in records if "n_bars=4.0" in record["path"]]

for i in range(len(records)):
    records[i]["bpm"] = records[i]["midi"].tempos[0].qpm

#%%
# remove 120 bpm
records_without_120 = [record for record in records if record["bpm"] != 120]

print(f"Original dataset size: {len(records)}")
print(f"Filtered dataset size: {len(records_without_120)}")

#%%

# print tempo distribution with histogram
import matplotlib.pyplot as plt
import numpy as np

tempos = [record["bpm"] for record in records]
plt.hist(tempos, bins=100)
plt.show()

# plot tempo distribution with histogram
tempos = [record["bpm"] for record in records_without_120]
plt.hist(tempos, bins=100)
#%%


sm_filter_fn = lambda sm : not any(track.program == 0 and not track.is_drum and "piano" not in track.name.lower() for track in sm.tracks)
# Filter out records containing program 0 non-drum tracks
filtered_records = [
    record
    for record in records
    if sm_filter_fn(record["midi"]) and record["bpm"] != 120
]

print(f"Original dataset size: {len(records)}")
print(f"Filtered dataset size: {len(filtered_records)}")
print(f"Removed {len(records) - len(filtered_records)} records")

#%%
# plot tempo histogram
tempos = [record["bpm"] for record in filtered_records]
plt.hist(tempos, bins=260, range=(40, 300))
#%%

track_meta = []

for record in records:
    midi = record["midi"]
    for track in midi.tracks:
        track_meta.append(
            {
                "program": track.program,
                "is_drum": track.is_drum,
                "name": track.name,
            }
        )
    
# %%
# for each progmram, get the top 10 most common track names if not drums
from collections import Counter

program_track_names = {}

for program in range(128):
    track_names = [
        track["name"]
        for track in track_meta
        if track["program"] == program and not track["is_drum"]
    ]
    program_track_names[program] = Counter(track_names).most_common(10)

for program, track_names in program_track_names.items():
    print(f"Program {program}")
    print(track_names)
    print()
    

#%% 

# print track names in order of occurence, ignore drums
track_names = [track["name"] for track in track_meta if not track["is_drum"]]

# export to text file, with each line containing a track name and its count, descending
with open("track_names.txt", "w") as f:
    for track_name, count in Counter(track_names).most_common():
        f.write(f"{track_name}: {count}\n")

#%%

# for files where the program is 0 and not drums, print the track names in order of occurence
for program_nr in [0,1,10,30]:
    track_names = []
    for record in records:
        midi = record["midi"]
        for track in midi.tracks:
            if track.program == program_nr and not track.is_drum:
                track_names.append(track.name)

    with open(f"track_names_program_{program_nr}_not_drums.txt", "w") as f:
        for track_name, count in Counter(track_names).most_common():
            f.write(f"{track_name}: {count}\n")

# %%
