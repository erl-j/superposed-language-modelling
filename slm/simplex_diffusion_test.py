#%%
device = "cuda:0"
from simplex_diffusion import SimplexDiffusionModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from util import preview_sm,piano_roll

# checkpoint = "../checkpoints/quiet-puddle-18/last.ckpt"
checkpoint = "../checkpoints/fanciful-planet-7/last.ckpt"
# checkpoint = "../checkpoints/super-mountain-5/last.ckpt"
# checkpoint = "../checkpoints/dauntless-aardvark-20/last.ckpt"
# checkpoint = "../checkpoints/twilight-haze-21/last.ckpt"
# checkpoint = "../checkpoints/zany-waterfall-23/last.ckpt"
checkpoint = "../checkpoints/effortless-resonance-33/last.ckpt"
# checkpoint = "../checkpoints/ethereal-disco-37/last.ckpt"
checkpoint = "../checkpoints/serene-sunset-44/last.ckpt"
# checkpoint = "../checkpoints/misunderstood-cloud-59/last.ckpt"
model = SimplexDiffusionModel.load_from_checkpoint(checkpoint, map_location=device)

# print model
print(model)
#%%

#%%
from data import MidiDataset
ROOT_DIR = "../"
TMP_DIR = ROOT_DIR + "artefacts/tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/output"

MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR+"data/mmd_loops/tst_midi_records_unique_pr.pt",
    path_filter_fn=lambda x: f"n_bars={MODEL_BARS}" in x,
    genre_list=model.tokenizer.config["tags"],
    tokenizer=model.tokenizer,
    min_notes=8 * MODEL_BARS,
    max_notes=model.tokenizer.config["max_notes"],
)


#%%
RESAMPLE_IDX = 1400

x = ds[RESAMPLE_IDX]
x_sm = model.tokenizer.decode(x)

preview_sm(x_sm)

#%%

# %%
sns.set_style("whitegrid", {'axes.grid' : False})


tokenizer = model.tokenizer

mask = tokenizer.constraint_mask(
    scale="C pentatonic",
    # tags=["metal"],
    # tempos=["126"],
    instruments = ["Drums","Bass","Guitar"],
    min_notes = 50,
    max_notes = 290,
    min_notes_per_instrument=10,
)

mask = tokenizer.infilling_mask(
    x=x,
    beat_range=(4, 12),
    min_notes=0,
    max_notes=290,
)


# beat_range=(0,16)
# pitch_range = [f"pitch:{i}" for i in range(30,108) ]+["pitch:-"]
# # make infilling mask
# mask = (
#     model.tokenizer.infilling_mask(
#         x,
#         beat_range,
#         min_notes=0,
#         max_notes=275,
#         pitches=pitch_range,
#         mode ="harmonic"
#     )[None, ...]    
#     .to(model.device)
#     .float()
# ) 


BATCH_SIZE = 2
N_STEPS = 100
y = model.sample(mask,
                 BATCH_SIZE,
                 N_STEPS,
                 device=device,
                 argmax=True,
                 temperature=1.0,
                 top_p=0.5,
                 mask_noise_factor = 5.0,
                 plot=False,
                 enforce_mask=True,
                 )

import matplotlib.pyplot as plt
import torch
y1h = torch.nn.functional.one_hot(y, num_classes=len(model.tokenizer.vocab)).float()

# no grid theme


plt.imshow(y1h[0].cpu().numpy().T, aspect="auto",interpolation="none")
plt.show()


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


preview_sm(y_sm)



# %%
