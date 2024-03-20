#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from slm.train import EncoderOnlyModel
from util import piano_roll
from data import MidiDataset
import torch
#%%

device = "cuda:7"

model = EncoderOnlyModel.load_from_checkpoint(
    "../checkpoints/eager-darkness-234/epoch=53-step=62370-val/loss_epoch=0.15.ckpt",
    map_location=device,
)

# Move the model to the device
model = model.to(device)


#%%


N_BARS = 4

genre_list = [
    "other",
    "pop",
    "rock",
    "italian%2cfrench%2cspanish",
    "classical",
    "romantic",
    "renaissance",
    "alternative-indie",
    "metal",
    "traditional",
    "country",
    "baroque",
    "punk",
    "modern",
    "jazz",
    "dance-eletric",
    "rnb-soul",
    "medley",
    "blues",
    "hip-hop-rap",
    "hits of the 2000s",
    "instrumental",
    "midi karaoke",
    "folk",
    "newage",
    "latino",
    "hits of the 1980s",
    "hits of 2011 2020",
    "musical%2cfilm%2ctv",
    "reggae-ska",
    "hits of the 1970s",
    "christian-gospel",
    "world",
    "early_20th_century",
    "hits of the 1990s",
    "grunge",
    "australian artists",
    "funk",
    "best of british",
]

# Load the dataset
val_ds = MidiDataset(
    cache_path="../artefacts/val_midi_records.pt",
    path_filter_fn=lambda x: f"n_bars={N_BARS}" in x,
    genre_list=genre_list,
    tokenizer=model.tokenizer,
    min_notes=8 * N_BARS,
    max_notes=model.tokenizer.config["max_notes"],
)


#%%

BATCH_SIZE = 64
# get val dataloader
val_dl = torch.utils.data.DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
)

# get one batch
x = next(iter(val_dl))

# move to device
x = x.to(device)

# get perofmrance curve
metrics = model.performance_curve(x)


# %%

# plot metrics
plt.plot(metrics)
plt.show()


# %%
