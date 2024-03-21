# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import piano_roll
import os
import IPython.display as ipd
import torch

#%%

device = "cuda:7"

ROOT_DIR = "../"

# ckpt = "checkpoints/exalted-cloud-246/epoch=89-step=103950-val/loss_epoch=0.14.ckpt"
ckpt = "checkpoints/desert-capybara-249/epoch=81-step=118326-val/loss_epoch=0.14.ckpt"
model = EncoderOnlyModel.load_from_checkpoint(
    ROOT_DIR + ckpt,
    map_location=device,
    avg_positional_encoding=True,
)
# Move the model to the device
model = model.to(device)


#%%
pos_z = model.positional_encoding[0].detach().cpu()[:model.tokenizer.config["max_notes"]]

print(pos_z.shape)

plt.figure(figsize=(10, 10))
sns.heatmap(pos_z.T, cmap="viridis")
plt.title("Positional Encoding")
plt.show()


#%%

N_BARS = 4

# Load the dataset
val_ds = MidiDataset(
    cache_path="../artefacts/val_midi_records.pt",
    path_filter_fn=lambda x: f"n_bars={N_BARS}" in x,
    genre_list=model.tokenizer.config["tags"],
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

for base_masking_ratio in np.linspace(0, 1, 5):
    # get perofmrance curve
    metrics = model.performance_curve(x, base_masking_ratio=base_masking_ratio)
    # plot metrics
    plt.plot(metrics)
    plt.title(f"Base Masking Ratio: {base_masking_ratio}")
    plt.show()


# %%


# %%


# %%




# %%
