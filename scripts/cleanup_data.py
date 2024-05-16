#%%
from data import MidiDataset

genre_list = ["pop"]

N_BARS = 4

trn_ds = MidiDataset(
    cache_path="../artefacts/val_midi_records_unique_pr.pt",
    path_filter_fn = lambda x: f"n_bars={N_BARS}" in x,
    genre_list=genre_list,
    tokenizer=None,
    transposition_range=[0,0],
)

# %%
sm = 
