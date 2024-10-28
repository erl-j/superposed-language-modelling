#%%

import symusic as sm
import torch

ckpt = "../../data/mmd_loops/val_midi_records_unique_pr.pt"

records = torch.load(ckpt)

# %%

from tqdm import tqdm
# record onset times, offset times and durations

onsets = []
offsets = []
durations = []

tpq = 24

for record in tqdm(records):
    for r in record:
        midi = r["midi"]
        midi = midi.resample(tpq)
        for track in midi.tracks:
            for note in track.notes:
                onsets.append(note.start)
                offsets.append(note.end)
                durations.append(note.duration)

#%%
ticks = 24 * 16
onset_counts = [0]  * ticks
offset_counts = [0] * ticks
duration_counts = [0] * ticks

for onset in onsets:
    if onset < ticks:
        onset_counts[onset] += 1

for offset in offsets:
    if offset < ticks:
        offset_counts[offset] += 1

for duration in durations:
    if duration < ticks:
        duration_counts[duration] += 1

#%%


# print offset times that happen less than 10 times
print("Offset times that happen less than 10 times")
for i, offset in enumerate(offset_counts):
    if offset < 100:
        print(i, offset)

# print onset times that happen less than 10 times
print("Onset times that happen less than 10 times")
for i, onset in enumerate(onset_counts):
    if onset < 100:
        print(i, onset)

# print duration times that happen less than 10 times
print("Duration times that happen less than 10 times")
for i, duration in enumerate(duration_counts):
    if duration < 100:
        print(i, duration)

#%%

print("Total notes: ", len(onsets))

# plot histogram of onset times
import matplotlib.pyplot as plt


plt.hist(onsets, bins=24*16)
plt.show()

# plot histogram of offset times
plt.hist(offsets, bins=24*16)
plt.show()

# plot histogram of durations
plt.hist(durations, bins=24*16)
plt.show()





# %%
