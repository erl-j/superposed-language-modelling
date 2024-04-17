#%%
device = "cuda:7"
from bfn import BFNModel

checkpoint = "../checkpoints/upbeat-dawn-53/epoch=1-step=3244-val/loss_epoch=0.00907.ckpt"

model = BFNModel.load_from_checkpoint(checkpoint, map_location=device)


#%%
from data import MidiDataset
ROOT_DIR = "../"
TMP_DIR = ROOT_DIR + "artefacts/tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/output"

MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR+"paper_assets/tst_midi_records_unique_pr.pt",
    path_filter_fn=lambda x: f"n_bars={MODEL_BARS}" in x,
    genre_list=model.tokenizer.config["tags"],
    tokenizer=model.tokenizer,
    min_notes=8 * MODEL_BARS,
    max_notes=model.tokenizer.config["max_notes"],
)


RESAMPLE_IDX = 50

x = ds[RESAMPLE_IDX]
x_sm = model.tokenizer.decode(x)





#%%
batch = x.unsqueeze(0).to(device)
model.preview_beta(batch)

# %%

print(model.beta1)

tokenizer = model.tokenizer

mask = tokenizer.constraint_mask(
    scale="C pentatonic",
    instruments = ["Piano","Drums","Bass"],
    min_notes = 50,
    max_notes = 100,
    min_notes_per_instrument=30,
)

BATCH_SIZE = 10
N_STEPS = 100
y = model.sample(None,BATCH_SIZE,N_STEPS,device=device,argmax=True)

import matplotlib.pyplot as plt
import torch
y1h = torch.nn.functional.one_hot(y, num_classes=len(model.tokenizer.vocab)).float()

plt.imshow(y1h[0].cpu().numpy().T, aspect="auto",interpolation="none")
plt.show()

from util import preview, piano_roll

# plot piano rolls,
# use a 16:9 aspect ratio for each plot
# subplots
fig, axs = plt.subplots(BATCH_SIZE,1, figsize=(4,2*BATCH_SIZE))
for i in range(BATCH_SIZE):        
    y_sm = model.tokenizer.decode(y[i])
    # print number of notes
    print(f"Number of notes: {y_sm.note_num()}")

    pr = piano_roll(y_sm, tpq=4)
    axs[i].imshow(pr, aspect="auto",interpolation="none")
plt.show()

#%%
# play audio of last 
preview(y_sm, tmp_dir="artefacts/tmp", audio=True)
    

# %%
