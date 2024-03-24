# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import piano_roll
import os
import IPython.display as ipd
from util import get_scale
from paper_checkpoints import SLM_CKPT_PTH, MLM_CKPT_PTH
import torch

# %%
device = "cuda:7"
ROOT_DIR = "../"

slm = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + SLM_CKPT_PTH,
        map_location=device,
        avg_positional_encoding=True,
    )
    .to(device)
    .eval()
)

mlm = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + MLM_CKPT_PTH,
        map_location=device,
        avg_positional_encoding=True,
    )
    .to(device)
    .eval()
)

#%%
N_BARS = 4
# Load the dataset
val_ds = MidiDataset(
    cache_path="../artefacts/val_midi_records.pt",
    path_filter_fn=lambda x: f"n_bars={N_BARS}" in x,
    genre_list=slm.tokenizer.config["tags"],
    tokenizer=None,
    min_notes=8 * N_BARS,
    max_notes=slm.tokenizer.config["max_notes"],
)
BATCH_SIZE = 64
# get val dataloader
val_dl = torch.utils.data.DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
)
#%%

for masking_ratio in np.linspace(0, 1, 5):
    for superposition_prob in np.linspace(0, 1, 5):
        superposition_ratio = 




for known_token_ratio in np.linspace(0, 1, 5):
    # get perforrance curve
    metrics = slm.performance_curve(val_dl, known_token_ratio=known_token_ratio)

    
    # plot metrics
    plt.plot(metrics)
    plt.title(f"Known Token Ratio: {known_token_ratio}")
    plt.show()



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
