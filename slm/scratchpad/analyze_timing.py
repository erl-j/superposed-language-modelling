#%%

import symusic as sm
import torch
import tempfile
import muspy
from tqdm import tqdm
ckpt = "../data/gmd_loops/trn_midi_records.pt"

records = torch.load(ckpt)


#%%
# count loops per record
loop_counts = [len([r for r in record if "n_bars=4" in r["path"]]) for record in records]

# plot histogram of loop counts
import matplotlib.pyplot as plt

plt.hist(loop_counts, bins=100)
plt.xlabel("Number of loops per song")
plt.ylabel("Number of songs")
plt.show()

# make cumulative histogram
plt.hist(loop_counts, bins=100, cumulative=True)
plt.xlabel("Number of loops per song")
plt.ylabel("Number of songs")
plt.show()

# i want to know how many loops are from a song with 1 loop, 2 loops, 3 loops, etc
# repeat loop_count[i] loop_count[i] times
loop_counts_bis = []
for i, count in enumerate(loop_counts):
    loop_counts_bis.extend([count] * count)

# make histogram of loop counts bis
plt.hist(loop_counts_bis, bins=100)
plt.xlabel("Number of loops per song loop came from")
plt.ylabel("Number of loops")
plt.show()


#%%

# flat the records
records = [r for record in records for r in record]

# count n_bars=1.0, 2.0 and 4.0. regex
import re

counts = [int(record["path"].split("n_bars=")[1][0]) for record in records]

# print counts
from collections import Counter

counter = Counter(counts)

for count, n in counter.items():
    print(f"n_bars={count}.0: {n}")



#%%

# preserve only those with "n_bars=4.0" in the path
records = [record for record in records if "n_bars=4.0" in record["path"]]

# Filter out records containing program 0 non-drum tracks
filtered_records = [
    record
    for record in records
    if not any(
        track.program == 0 and not track.is_drum for track in record["midi"].tracks
    )
]

print(f"Original dataset size: {len(records)}")
print(f"Filtered dataset size: {len(filtered_records)}")
print(f"Removed {len(records) - len(filtered_records)} records")

tempos = []
for record_idx, record in enumerate(tqdm(records)):
    record = records[record_idx]
    tempos.append(record["midi"].tempos[0].qpm)
    # dump to tmpfile
    with tempfile.NamedTemporaryFile(suffix=".mid") as f:
        record["midi"].dump_midi(f.name)
        muspy_midi = muspy.read_midi(f.name)
    pis_rate = muspy.scale_consistency(muspy_midi)
    # add pis rate to record
    records[record_idx]["pis_rate"] = pis_rate

#%% 

# histogram of pis rates
import matplotlib.pyplot as plt

plt.hist([record["pis_rate"] for record in records], bins=100)
plt.show()

#%%
# preview 10 midi files with the worst pis rates
from util import preview_sm
import numpy as np

# remove nan records
records = [record for record in records if not np.isnan(record["pis_rate"])]

#%%

# get all track names and program numbers

track_meta = []

for record in records:
    midi = record["midi"]
    for track in midi.tracks:
        track_meta.append(({"name": track.name, "program": track.program, "is_drum": track.is_drum}))


#%%

# for each program number, show 10 most common track names
from collections import Counter

program_track_name_counter = Counter((track["program"], track["name"]) for track in track_meta)

for (program, track_name), count in program_track_name_counter.most_common():
    print(program, track_name, count)



#%%

# print track names by count
from collections import Counter

track_name_counter = Counter(track["name"] for track in track_meta)

for track_name, count in track_name_counter.most_common():
    print(track_name, count)


# print program numbers by count
program_counter = Counter(track["program"] for track in track_meta)

#%%
for program, count in program_counter.most_common():
    print(program, count)


# non drum program counter

#%%

non_drum_program_counter = Counter(track["program"] for track in track_meta if not track["is_drum"])

for program, count in non_drum_program_counter.most_common():
    print(program, count)


#%%

worst_records = sorted(records, key=lambda x: x["pis_rate"])

for record in worst_records[:100]:
    preview_sm(record["midi"])
    print(record["pis_rate"])


#%% compute bpm

records = [{**record, "bpm": record["midi"].tempos[0].qpm} for record in records]

# now scatter plot bpm vs pis rate

#%%
piano_tracks = []
drum_tracks = []
piano_named_tracks = []

for record in tqdm(records):
    for track in record["midi"].tracks:
        if track.program==0 and not track.is_drum:
            print(track.name)
            piano_tracks.append(track)
        elif track.is_drum:
            drum_tracks.append(track)

        if track.name == "Piano":
            piano_named_tracks.append(track)

        

#%% get number of distinct pitches

n_distinct_pitches_in_piano_tracks = [len(set(note.pitch for note in track.notes)) for track in piano_tracks]
piano_range_size = [max(note.pitch for note in track.notes) - min(note.pitch for note in track.notes) for track in piano_tracks]

n_distinct_pitches_in_drum_tracks = [len(set(note.pitch for note in track.notes)) for track in drum_tracks]
drum_range_size = [max(note.pitch for note in track.notes) - min(note.pitch for note in track.notes) for track in drum_tracks]

n_distinct_pitches_in_named_piano_tracks = [len(set(note.pitch for note in track.notes)) for track in piano_named_tracks]
named_piano_range_size = [max(note.pitch for note in track.notes) - min(note.pitch for note in track.notes) for track in piano_named_tracks]

#%%

# make scatter plot of n_distinct vs range

import matplotlib.pyplot as plt

plt.scatter(n_distinct_pitches_in_piano_tracks, piano_range_size, alpha=0.1, label="piano")
plt.scatter(n_distinct_pitches_in_drum_tracks, drum_range_size, alpha=0.1, label="drum")
plt.scatter(n_distinct_pitches_in_named_piano_tracks, named_piano_range_size, alpha=0.1, label="named piano")
plt.legend()
plt.xlabel("n distinct pitches")
plt.ylabel("range size")
plt.show()


# plot histogram distinct pitches, with both piano and drum tracks
# import matplotlib.pyplot as plt

# plt.hist(n_distinct_pitches_in_piano_tracks, bins=100, range=(0, 40))
# plt.show()

# plt.hist(n_distinct_pitches_in_drum_tracks, bins=100, range=(0, 40))
# plt.show()

# plt.hist(n_distinct_pitches_in_named_piano_tracks, bins=100, range=(0, 40))
# plt.show()





# plt.hist(piano_range_size, bins=100, range=(0, 40))

# plt.show()


# plt.hist(drum_range_size, bins=100, range=(0, 40))

# plt.show()

# plt.hist(named_piano_range_size, bins=100, range=(0, 40))

# plt.show()



#%%
#%%
import matplotlib.pyplot as plt

plt.scatter([record["bpm"] for record in records], [record["pis_rate"] for record in records], alpha=0.1, s=1)
plt.show()

# for each interval of bpm, show the pis histogram
bpm_intervals = [(i, i+10) for i in range(50, 200, 10)]

for bpm_interval in bpm_intervals:
    bpm_min, bpm_max = bpm_interval
    records_in_interval = [record for record in records if bpm_min <= record["bpm"] < bpm_max]
    pis_rates = [record["pis_rate"] for record in records_in_interval]
    plt.hist(pis_rates, bins=100)
    plt.title(f"bpm: {bpm_min}-{bpm_max}")
    plt.show()


#%%


# best_records = sorted(records, key=lambda x: x["pis_rate"], reverse=True)

# for record in best_records[:10]:
#     preview_sm(record["midi"])
#     print(record["pis_rate"])

#%%
# histogram of pis rates
pis_rates = [record["pis_rate"] for record in records]


import matplotlib.pyplot as plt

plt.hist(pis_rates, bins=100)
plt.show()
#%%

# make histogram of tempos
import matplotlib.pyplot as plt

plt.hist(tempos, bins=100)
plt.show()

#%%
# get the most common tempo



#%%

from tqdm import tqdm
# record onset times, offset times and durations

onsets = []
offsets = []
durations = []

tpq = 96

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
