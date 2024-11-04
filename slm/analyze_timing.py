#%%

import symusic as sm
import torch

ckpt = "../data/mmd_loops/val_midi_records_unique_pr.pt"

records = torch.load(ckpt)

# %%

from tqdm import tqdm
# record onset times, offset times and durations

onsets = []
offsets = []
durations = []

tpq = 12

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

# get max onset
max_onset = tpq * 16
max_offset = tpq * 16
max_duration = tpq * 16

onset_counts_per_tick = [0] * (max_onset+1)

for onset in onsets:
    if onset < max_onset:
        onset_counts_per_tick[onset] += 1

offset_counts_per_tick = [0] * (max_offset+1)

for offset in offsets:
    if offset < max_offset:
        offset_counts_per_tick[offset] += 1

duration_counts_per_tick = [0] * (max_duration+1)
for duration in durations:
    if duration < max_duration:
        duration_counts_per_tick[duration] += 1


#%%

# use bars
import matplotlib.pyplot as plt
import numpy as np

# log scale
# make figure wide
plt.figure(figsize=(20, 10))

# do not show line
plt.bar(
    np.arange(max_onset+1),
    onset_counts_per_tick,
    width=1,
    align="edge"
)
# add big tick every beat (tpq ticks)
plt.xticks(
    np.arange(0, max_onset+1, tpq),
    np.arange(0, max_onset+1, tpq) / tpq
)

plt.show()

#%%
plt.figure(figsize=(20, 10))
plt.plot(
    offset_counts_per_tick)

plt.show()

plt.figure(figsize=(20, 10))
plt.plot(
    duration_counts_per_tick)
plt.show()

#%%

#%%


# show ticks in order of frequency
# count the number of times each tick appears in the onsets, offsets and durations
# and print the ticks in order of frequency
# this will help us identify the most common ticks

from collections import Counter

onset_counter = Counter(onsets)
offset_counter = Counter(offsets)
duration_counter = Counter(durations)

print("Onset times in order of frequency")
for onset, count in onset_counter.most_common():
    print(onset, count)

print("Offset times in order of frequency")
for offset, count in offset_counter.most_common():
    print(offset, count)

print("Duration times in order of frequency")
for duration, count in duration_counter.most_common():
    print(duration, count)


#%%
ticks = tpq * 16
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
