#%%
device = "cuda:0"
from simplex_diffusion import SimplexDiffusionModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from util import preview_sm,piano_roll
import matplotlib.pyplot as plt
import seaborn as sns
from data import MidiDataset
import torch

checkpoint = "../checkpoints/fanciful-planet-7/last.ckpt"
# checkpoint = "../checkpoints/super-mountain-5/last.ckpt"
checkpoint = "../checkpoints/effortless-resonance-33/last.ckpt"
checkpoint = "../checkpoints/serene-sunset-44/last.ckpt"
# checkpoint = "../checkpoints/driven-violet-62/last.ckpt"
model = SimplexDiffusionModel.load_from_checkpoint(checkpoint, map_location=device)

# print model
#%%

# get the embedding
embedding = model.embedding_layer.weight.detach().cpu().numpy().T
projection = model.decoder_output_layer.weight.detach().cpu().numpy()

# get vocab
vocab = model.tokenizer.vocab

embedding_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
projection_norm = np.linalg.norm(projection, axis=0, keepdims=True)

# normalize the embedding
embedding = embedding / embedding_norm
projection = projection / projection_norm

# compute the cosine similarity
embedding_similarity = embedding @ embedding.T
projection_similarity = projection @ projection.T


# plot the similarity matrix
# make large figure
plt.figure(figsize=(50, 50))
# set small font
sns.set(font_scale=0.5)
sns.heatmap(embedding_similarity, cmap="magma", xticklabels=vocab, yticklabels=vocab, mask = np.eye(len(vocab)))
plt.show()

# plot the similarity matrix
# make large figure
plt.figure(figsize=(50, 50))
# set small font
sns.set(font_scale=0.5)
sns.heatmap(projection_similarity, cmap="magma", xticklabels=vocab, yticklabels=vocab, mask = np.eye(len(vocab)))
plt.show()

#%%
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
x2 = ds[RESAMPLE_IDX+500]
x_sm2 = model.tokenizer.decode(x2)

preview_sm(x_sm2)
#%%

# %%
sns.set_style("whitegrid", {'axes.grid' : False})


tokenizer = model.tokenizer

mask = tokenizer.constraint_mask(
    scale="C pentatonic",
    tags=["pop"],
    tempos=["126"],
    instruments = ["Piano","Bass","Drums"],
    min_notes = 50,
    max_notes = 250,
    min_notes_per_instrument=40,
)


# mask = tokenizer.infilling_mask(
#     x=x,
#     beat_range=(4, 12),
#     min_notes=0,
#     max_notes=275,
# )


# beat_range=(0,16)
# pitch_range = [f"pitch:{i}" for i in range(50,108) ]+["pitch:-"]
# #make infilling mask
# mask = (
#     model.tokenizer.infilling_mask(
#         x,
#         beat_range,
#         min_notes=x_sm.note_num(),
#         max_notes=x_sm.note_num(),
#         pitches=pitch_range,
#         mode ="harmonic"
#     )[None, ...]    
#     .to(model.device)
#     .float()
# ) 

# import torch
# mask = torch.nn.functional.one_hot(x, num_classes=len(model.tokenizer.vocab)).float()
# # mask2 = torch.nn.functional.one_hot(x2, num_classes=len(model.tokenizer.vocab)).float()

# mask = mask.mean(dim=0, keepdim=True) * torch.ones_like(mask)
# plt.imshow(mask.cpu().numpy().T, aspect="auto",interpolation="none")
# plt.show()


format_mask = torch.tensor(tokenizer.get_format_mask(), device=model.device).float()

mask = mask.to(model.device).float()


plt.imshow(mask.cpu().numpy().T, aspect="auto",interpolation="none")
plt.show()

plt.imshow(format_mask.cpu().numpy().T, aspect="auto",interpolation="none")
plt.show()

plt.imshow((mask*format_mask).cpu().T, aspect="auto",interpolation="none")
plt.show()
# assert torch.allclose(mask, mask*format_mask)

mask = mask * format_mask


# set torch seed
torch.manual_seed(0)
# infilling top-p 0.5
BATCH_SIZE = 2
N_STEPS = 30

prior = mask / mask.sum(dim=-1, keepdim=True)
# check that the prior is normalized
assert torch.allclose(prior.sum(dim=-1), torch.ones_like(prior.sum(dim=-1)))
plt.imshow(prior.cpu().numpy().T, aspect="auto",interpolation="none")
plt.show()
y = model.sample2(prior,
                BATCH_SIZE,
                N_STEPS,
                device=device,
                argmax=True,
                temperature=1.0,
                top_p=0.5,
                prior_strength = 1.0,
                plot=False,
                post_prior=True
                )


# y = model.sample(mask,
#                  BATCH_SIZE,
#                  N_STEPS,
#                  device=device,
#                  argmax=True,
#                  temperature=1.0,
#                  top_p=0.0,
#                  mask_noise_factor = 5.0,
#                  plot=True,
#                  enforce_mask=True,
#                  )

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
