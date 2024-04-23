#%%
device = "cuda:1"
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
model = SimplexDiffusionModel.load_from_checkpoint(checkpoint, map_location=device)

# print model
print(model)
#%%
import seaborn as sns

#%% no grid

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


cmap = "magma"
# plot the similarity matrix
# make large figure
plt.figure(figsize=(50, 50))
# set small font
sns.set(font_scale=0.5)
sns.heatmap(embedding_similarity, cmap=cmap, xticklabels=vocab, yticklabels=vocab, mask = np.eye(len(vocab)),vmax=1,vmin=-1)
plt.show()

# plot the similarity matrix
# make large figure
plt.figure(figsize=(50, 50))
# set small font
sns.set(font_scale=0.5)
sns.heatmap(projection_similarity, cmap=cmap, xticklabels=vocab, yticklabels=vocab, mask = np.eye(len(vocab)),vmax=1,vmin=-1)
plt.show()

#%%

# per attribute embedding_similarity
for attr in model.tokenizer.note_attribute_order:

    tokens = [token for token in vocab if attr+":" in token]
    indices = [model.tokenizer.token2idx[token] for token in tokens]
    # get the indices of the attribute
    # get the embedding
    emb = embedding[indices,:]

    pro = projection[indices,:]

    # compute the cosine embedding_similarity
    embedding_similarity = emb @ emb.T

    projection_similarity = pro @ pro.T

    # plot the embedding_similarity matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(embedding_similarity, cmap=cmap, xticklabels=tokens, yticklabels=tokens, mask = np.eye(len(tokens)))
    plt.title(attr)
    plt.show()

    # plot the embedding_similarity matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(projection_similarity, cmap=cmap, xticklabels=tokens, yticklabels=tokens, mask = np.eye(len(tokens)))
    plt.title(attr)
    plt.show()

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
RESAMPLE_IDX = 130

x = ds[RESAMPLE_IDX]
x_sm = model.tokenizer.decode(x)

preview_sm(x_sm)

#%%

# %%
sns.set_style("whitegrid", {'axes.grid' : False})


tokenizer = model.tokenizer

# mask = tokenizer.constraint_mask(
#     # scale="C pentatonic",
#     # tags=["metal"],
#     # tempos=["126"],
#     instruments = ["Drums","Bass",],
#     min_notes = 50,
#     max_notes = 150,
#     min_notes_per_instrument=50,
# )

mask = tokenizer.infilling_mask(
    x=x,
    beat_range=(0, 8),
    min_notes=100,
    max_notes=290,
)


beat_range=(0,16)
pitch_range = [f"pitch:{i}" for i in range(50,108) ]+["pitch:-"]
# make infilling mask
mask = (
    model.tokenizer.infilling_mask(
        x,
        beat_range,
        min_notes=0,
        max_notes=290,
        pitches=pitch_range,
    )[None, ...]    
    .to(model.device)
    .float()
) 

BATCH_SIZE = 2
N_STEPS = 100
y = model.sample(mask,
                 BATCH_SIZE,
                 N_STEPS,
                 device=device,
                 argmax=True,
                 temperature=1.0,
                 top_p=0.9,
                 mask_noise_factor = 2.5,
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
