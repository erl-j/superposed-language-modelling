#%%
device = "cuda:5"
from bfn import BFNModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from util import preview_sm

# checkpoint = "../checkpoints/valiant-butterfly-54/last.ckpt"
# checkpoint ="../checkpoints/still-universe-56/last.ckpt"
# checkpoint = "../checkpoints/sparkling-paper-58/last.ckpt"
# checkpoint ="../checkpoints/worldly-music-61/last.ckpt"
# checkpoint = "../checkpoints/kind-oath-66/last.ckpt"
# checkpoint = "../checkpoints/golden-capybara-67/last.ckpt"
# checkpoint = "../checkpoints/avid-durian-68/last.ckpt"
# checkpoint = "../checkpoints/ethereal-star-75/last.ckpt"
checkpoint = "../checkpoints/crisp-surf-78/last.ckpt"
checkpoint = "../checkpoints/polished-pine-82/last.ckpt"
#checkpoint = "../checkpoints/fallen-resonance-102/last.ckpt"
# checkpoint = "../checkpoints/fallen-resonance-102/last.ckpt"
# checkpoint = "../checkpoints/major-wood-115/last.ckpt"
# checkpoint = "../checkpoints/azure-surf-114/last.ckpt"
checkpoint = "../checkpoints/deft-puddle-113/last.ckpt"
# checkpoint = "../checkpoints/stellar-breeze-112/last.ckpt"
# checkpoint = "../checkpoints/worthy-planet-105/last.ckpt"
# checkpoint = "../checkpoints/feasible-violet-110/last.ckpt"
checkpoint = "../checkpoints/polished-dream-124/last.ckpt"
# checkpoint = "../checkpoints/skilled-wave-136/last.ckpt"

# checkpoint = "../checkpoints/light-deluge-138/last.ckpt"

# checkpoint = "../checkpoints/devout-cherry-140/last.ckpt"
device="cuda:0"

model = BFNModel.load_from_checkpoint(checkpoint, map_location="cpu").to(device)

# print model
print(model)


#%%
# repeat in first dim
batch = model.format_mask[None,:].repeat(10,1,1).argmax(-1)

# plot beta curve
model.preview_beta(batch)

#%% no grid

# get the embedding
embedding = model.embedding_layer.weight.detach().cpu().numpy().T
projection = model.decoder_output_layer.weight.detach().cpu().numpy()

# get vocab
vocab = model.tokenizer.vocab

embedding_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
projection_norm = np.linalg.norm(projection, axis=1, keepdims=True)

# normalize the embedding
embedding = embedding / embedding_norm
projection = projection / projection_norm

# compute the cosine similarity
embedding_similarity = embedding @ embedding.T
projection_similarity = projection @ projection.T


cmap = "Spectral_r"
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

dataset = "mmd_loops"

MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR+f"data/{dataset}/tst_midi_records_unique_pr.pt",
    path_filter_fn=lambda x: f"n_bars={MODEL_BARS}" in x if dataset=="mmd_loops" else True,
    genre_list=model.tokenizer.config["tags"],
    tokenizer=model.tokenizer,
    min_notes=8 * MODEL_BARS,
    max_notes=model.tokenizer.config["max_notes"],
)

#%%
from util import preview_sm
RESAMPLE_IDX = 1400

x = ds[RESAMPLE_IDX]
x_sm = model.tokenizer.decode(x)
preview_sm(x_sm)

#%%

sns.set_style("whitegrid", {'axes.grid' : False})

tokenizer = model.tokenizer

# mask = tokenizer.constraint_mask(
#     # scale="C pentatonic",
#     tags = ["metal"],
#     instruments = ["Drums","Bass","Guitar"],
#     min_notes = 1,
#     max_notes = 299,
#     min_notes_per_instrument=20,
#     tempos=["138"]
# )

# mask = model.tokenizer.infilling_mask(
#     x=x,
#     # beat_range=(4, 15),
#     pitches=[f"pitch:{i}" for i in range(49, 100)],
#     # min_notes=10,
#     # max_notes=250,
#     min_notes=x_sm.note_num(),
#     max_notes=x_sm.note_num(),
#     mode="harmonic"
# ).float()


mask =  model.tokenizer.replace_instruments_mask(
    x=x,
    instruments_to_remove=["Bass"],
    instruments_to_add=["Bass"],
    min_notes_per_instrument=20,
)


# import symusic as sm
# from util import preview, piano_roll, preview_sm
# import torch

# x_sm = sm.Score("../data/bob.mid")

# def crop_sm(x_sm, start_beat, end_beat):
#     tpq = x_sm.tpq
#     start = start_beat * tpq
#     end = end_beat * tpq
#     x_sm = x_sm.copy()
#     for track in x_sm.tracks:
#         track.notes = [note for note in track.notes if note.start >= start and note.end <= end]
#     return x_sm

# x_sm = crop_sm(x_sm, 0, 16)

# x = torch.tensor(model.tokenizer.encode(x_sm,tag="traditional"))

# preview_sm(x_sm)


# mask = model.tokenizer.replace_instruments_mask(
#     x=x,
#     instruments_to_remove=[],
#     instruments_to_add=["Drums","Bass","Guitar"],
#     min_notes_per_instrument=10,
# )

mask = mask * model.format_mask
prior = (mask / mask.sum(-1, keepdim=True)).float()

BATCH_SIZE = 2
N_STEPS = 100
TEMPERATURE = 0.7

y = model.sample(prior,BATCH_SIZE,N_STEPS, temperature=TEMPERATURE ,device=device,argmax=True, plot_interval=-1)

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

    # pr = piano_roll(y_sm, tpq=4)
    # axs[i].imshow(pr, aspect="auto",interpolation="none")
plt.show()

# play audio of last 
print(f"Number of notes: {y_sm.note_num()}")
preview_sm(y_sm)
    

# %%
