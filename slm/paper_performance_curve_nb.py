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
import random

#%%

device = "cuda:7"

ROOT_DIR = "../"

# ckpt = "checkpoints/exalted-cloud-246/epoch=89-step=103950-val/loss_epoch=0.14.ckpt"
# ckpt = "checkpoints/clear-terrain-265/epoch=111-step=161616-val/loss_epoch=0.14.ckpt"

mlm_ckpt = "checkpoints/dry-shadow-267/epoch=2-step=4329-val/loss_epoch=1.72.ckpt"
model = EncoderOnlyModel.load_from_checkpoint(
    ROOT_DIR + mlm_ckpt,
    map_location=device,
)

N_BARS = 4


# Move the model to the device
model = model.to(device)

model.eval()

# Load the dataset
val_ds = MidiDataset(
    cache_path="../artefacts/val_midi_records.pt",
    path_filter_fn=lambda x: f"n_bars={N_BARS}" in x,
    genre_list=model.tokenizer.config["tags"],
    tokenizer=model.tokenizer,
    min_notes=8 * N_BARS,
    max_notes=model.tokenizer.config["max_notes"],
)

BATCH_SIZE = 64
# get val dataloader
val_dl = torch.utils.data.DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
)


#%%

batch = next(iter(val_dl)).to(device)

masking_ratios = np.linspace(0.1, 1, 5)

superposition_density = np.linspace(0.1, 1, 5)

seed = 0

# set seed
torch.manual_seed(seed)

losses = []

max_notes = model.tokenizer.config["max_notes"]
n_attributes = len(model.tokenizer.note_attribute_order)

seq_len = max_notes * n_attributes


for masking_ratio in masking_ratios:

    x = batch.clone() 
    x = torch.nn.functional.one_hot(x, len(model.tokenizer.vocab)).float()

    x_target_idx = torch.argmax(x, dim=-1)
    
    # create mask for the input
    mask_noteattr_idx = random.sample( 
        range(seq_len),
        k= int(seq_len * masking_ratio)
    )

    x_masked = x

    x_masked[:, mask_noteattr_idx, :] = 1
   
    # get the output
    with torch.no_grad():
        logits = model(x_masked)

        # get masked indices

        masked_logits = logits[:, mask_noteattr_idx, :]
        masked_target_idx = x_target_idx[:, mask_noteattr_idx]


        # get cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            masked_logits.reshape(-1, len(model.tokenizer.vocab)),
            masked_target_idx.flatten()
        )

        losses.append({
            "masking_ratio": masking_ratio,
            "loss": loss.item()
        })

    print(f"Masking Ratio: {masking_ratio:.2f}, Loss: {loss.item()}")


#%%
# plot masking ratio on x and loss on y
import pandas as pd

losses = pd.DataFrame(losses)

plt.plot(losses["masking_ratio"], losses["loss"])
plt.xlabel("Masking Ratio")
plt.ylabel("Cross Entropy Loss")
plt.title("Masking Ratio vs Loss")

#%%




        





        

    

        



    # get the output
    




    



#%%
pos_z = model.positional_encoding[0].detach().cpu()[:model.tokenizer.config["max_notes"]]

print(pos_z.shape)

plt.figure(figsize=(10, 10))
sns.heatmap(pos_z.T, cmap="viridis")
plt.title("Positional Encoding")
plt.show()


#%%


#%%




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
