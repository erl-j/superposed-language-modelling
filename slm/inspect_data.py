#%%
from data import MidiDataset

genre_list = ["pop"]

N_BARS = 4



trn_ds = MidiDataset(
    cache_path="../artefacts/trn_midi_records.pt",
    path_filter_fn = lambda x: f"n_bars={N_BARS}" in x,
    genre_list=genre_list,
    tokenizer=None,
    transposition_range=[-4, 4],
)
# %%


# flatten the records
records = [x for sublist in trn_ds.records for x in sublist]

#%%
# get n_notes in midi
n_notes = [sum([track.note_num() for track in x["midi"].tracks if not track.name.startswith("Layer")]) for x in records]



        
# %%

import matplotlib.pyplot as plt

# show the distribution of n_notes
plt.hist(n_notes, range(0, 500, 1), alpha=0.75, label='n_notes')
plt.show()

# %%
