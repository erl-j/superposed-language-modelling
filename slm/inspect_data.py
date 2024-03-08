#%%
from data import MidiDataset

genre_list = ["pop"]

N_BARS = 4



trn_ds = MidiDataset(
    cache_path="../artefacts/val_midi_records.pt",
    path_filter_fn = lambda x: f"n_bars={N_BARS}" in x,
    genre_list=genre_list,
    tokenizer=None,
    transposition_range=[-4, 4],
)
# %%


# flatten the records
records = [x for sublist in trn_ds.records for x in sublist]


#%%
# get velocities

all_velocities = [[[note.velocity for note in track.notes] for track in record["midi"].tracks if not track.name.startswith("Layer")] for record in records]

# flatten the velocities
all_velocities = [x for sublist in all_velocities for x in sublist]
# flatten the velocities
all_velocities = [x for sublist in all_velocities for x in sublist]

#%%
# get all program numbers
all_programs = [[track.program for track in record["midi"].tracks if not track.name.startswith("Layer")] for record in records]
all_programs = [x for sublist in all_programs for x in sublist]

# pitches

#%%

all_pitches = [[[note.pitch for note in track.notes] for track in record["midi"].tracks if not track.name.startswith("Layer")] for record in records]
all_pitches = [x for sublist in all_pitches for x in sublist]
all_pitches = [x for sublist in all_pitches for x in sublist]

# histogram of pitches
import matplotlib.pyplot as plt

plt.hist(all_pitches, range(0, 128, 1), alpha=0.75, label='pitches')
plt.show()


#%%
# log scale
plt.hist(all_pitches, range(0, 128, 1), alpha=0.75, label='pitches', log=True)
plt.show()

#%%
# print min and max
print(max(all_programs))
print(min(all_programs))

#%%
# print max

print(max(all_velocities))
print(min(all_velocities))

#%%
print(max(all_programs))
print(min(all_programs))
#%%
# show histogram of velocities
import matplotlib.pyplot as plt

plt.hist(all_velocities, range(0, 128, 1), alpha=0.75, label='velocities')
plt.show()

# now with log scale
plt.hist(all_velocities, range(0, 128, 1), alpha=0.75, label='velocities', log=True)
plt.show()

# bins 


#%%
# get n_notes in midi
n_notes = [sum([track.note_num() for track in x["midi"].tracks if not track.name.startswith("Layer")]) for x in records]



        
# %%

import matplotlib.pyplot as plt

# show the distribution of n_notes
plt.hist(n_notes, range(0, 500, 1), alpha=0.75, label='n_notes')
plt.show()

# %%
