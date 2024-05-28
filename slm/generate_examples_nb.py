#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import preview_sm
import os
import IPython.display as ipd
from paper_checkpoints import checkpoints
import torch

device = "cuda:7"
ROOT_DIR = "../"

MODEL = "slm"

OUTPUT_DIR = ROOT_DIR + "artefacts/examples_4"
TMP_DIR = ROOT_DIR + "artefacts/tmp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + checkpoints[MODEL],
        map_location=device,
    )
    .to(device)
    .eval()
)

# if MODEL == "mlm":
#     model.mlm_restricted_sampling = True
#%%


#%%

a = model.format_mask[None, ...].to(model.device)
c = model.tokenizer.constraint_mask(
    tags=["classical"],
    # tags=["other"],
    instruments=["Piano","Ensemble","Pipe"],
    # tempos=["138"],
    pitches=["pitch:{i}" for i in range(60, 108)],
    scale="G major",
    tempos=["128"],
    min_notes=10,
    max_notes=290,
    min_notes_per_instrument=20
)[None, ...].to(model.device)
a = c * a

# Generate a sequence
y = model.generate(
    a,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
)[0].argmax(axis=1)

# y = model.generate_gibbs(
#     a,
#     temperature=1.0,
#     steps=300,
#     top_p=1.0,
#     top_k=0,
#     pmax=0.9,
#     pmin=0.001,
#     alpha=0.1,
# )[0].argmax(axis=1)


# decode
y_sm = model.tokenizer.decode(y)

print(f"Number of notes: {y_sm.note_num()}")

preview_sm(y_sm)

from util import sm_fix_overlap_notes

y_sm = sm_fix_overlap_notes(y_sm)

y_sm.dump_midi(OUTPUT_DIR + "/pitch_set_constraint_c.mid")

#%%

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


RESAMPLE_IDX = 1400

x = ds[RESAMPLE_IDX]
x_sm = model.tokenizer.decode(x)

preview_sm(x_sm)

x_sm.dump_midi(OUTPUT_DIR + "/resample_original.mid")

#%%

mask = model.tokenizer.infilling_mask(
    x=x,
    beat_range=(4, 15),
    min_notes=10,
    max_notes=250,
)[None,...].float().to(model.device)

plt.figure(figsize=(10,10))
plt.imshow(mask[0].cpu().numpy().T, aspect="auto",interpolation="nearest")
plt.show()

y = model.generate(
        mask,
        temperature=0.99,
        top_p=1.0,
        top_k=0,
        order = "random"
    )[0].cpu().numpy().argmax(axis=-1)

y_tokens = model.tokenizer.indices_to_tokens(y)

print(y_tokens)

y_sm = model.tokenizer.decode(y)

print(f"Number of notes: {y_sm.note_num()}")

preview_sm(y_sm)

#%%
from util import preview_sm

preview_sm(y_sm)

for track in y_sm.tracks:
    print(track.name)
print("\n")
for track in x_sm.tracks:
    print(track.name)

#%%

# one hot encode x
x1h = torch.nn.functional.one_hot(x, len(model.tokenizer.vocab)).float().to(model.device)

# sum across the note dimension
mask= (x1h.sum(axis=0)>0).float().repeat(x1h.shape[0],1)

# multiply by the format mask
mask = mask * model.format_mask[None, ...].to(model.device).float()

plt.figure(figsize=(10,10))
plt.imshow(mask.cpu().numpy().T, aspect="auto",interpolation="nearest")
plt.show()


# use as mask

y = model.generate(
        mask,
        temperature=1.5,
        top_p=1.0,
        top_k=0,
        order = "random"
    )[0].cpu().numpy().argmax(axis=-1)

y_sm = model.tokenizer.decode(y)

print(f"Number of notes: {y_sm.note_num()}")

preview(y_sm, TMP_DIR)

for track in y_sm.tracks:
    print(track.name)
print("\n")
for track in x_sm.tracks:
    print(track.name)


#%%
x_sm = model.tokenizer.decode(x)

# beat range
# beat_range=(8,12)
beat_range = (8, 12)


# make infilling mask
mask = (
    model.tokenizer.infilling_mask(
        x,
        beat_range=beat_range,
        # min_notes = 0,
        # max_notes = 290
        # min_notes=0,
        # max_notes=290,
        min_notes=x_sm.note_num(),
        max_notes=x_sm.note_num(),
    )[None, ...]
    .to(model.device)
    .float()
)

# y = (
#     model.generate(
#         mask,
#         temperature=0.9,
#         top_p=1.0,
#         top_k=0,
#         order = "random"
#     )[0]
#     .cpu()
#     .numpy()
#     .argmax(axis=-1)
# )

y = model.generate_gibbs(
    mask, 
    temperature=0.99,
    top_p=1.0,
    top_k=0,
    steps=1000,
    pmax = 0.5,
    pmin = 0.1,
    alpha=0.7,
)[0].argmax(axis=1)



# y = (
#     model.generate(
#         mask, 
#         temperature=0.85, 
#         top_p=1.0, 
#         top_k=0,
#         typical_sampling_t=-1,
#         order = "random",
#         temperature_decay=False,
#         min_temperature=0,
#         )[0]
#     .cpu()
#     .numpy()
#     .argmax(axis=-1)
# )



y_sm = model.tokenizer.decode(y)

print(f"Number of notes: {y_sm.note_num()}")


preview(y_sm, TMP_DIR)
y_sm.dump_midi(OUTPUT_DIR + "/infilling_middle.mid")

#%%

mask = model.tokenizer.replace_mask(x, ["pitch"]).to(model.device).float()

y = model.generate(
    mask,
    temperature=1.0,
    top_p=1,
)[0].cpu().numpy().argmax(axis=-1)

# convert to tokens
tokens = model.tokenizer.indices_to_tokens(y)

y_sm = model.tokenizer.decode(y)

preview(y_sm, TMP_DIR)

y_sm.dump_midi(OUTPUT_DIR + "/slm_replace_pitch.mid")



#%% 

# replace pitch with pitch set constraint

# bug with drums?

mask = model.tokenizer.replace_mask(x, ["pitch"]).to(model.device).float()

mask2 = model.tokenizer.constraint_mask(
    scale="G pentatonic",
    min_notes=0,
)[None, ...].to(model.device).float()

mask = mask * mask2

y = (
    model.generate(
        mask,
        temperature=1.0,
        top_p=1,
        top_k=0,
    )[0]
    .cpu()
    .numpy()
    .argmax(axis=-1)
)

y_sm = model.tokenizer.decode(y)

preview(y_sm, TMP_DIR)

y_sm.dump_midi(OUTPUT_DIR + "/slm_replace_pitch_pentatonic.mid")


#%%

mask = model.tokenizer.replace_mask(x, ["instrument"]).to(model.device).float()

y = (
    model.generate(
        mask,
        temperature=1.5,
        schedule_fn=lambda x: x,
        top_p=1,
        top_k=0,
    )[0]
    .cpu()
    .numpy()
    .argmax(axis=-1)
)

y_sm = model.tokenizer.decode(y)

preview(y_sm, TMP_DIR)

y_sm.dump_midi(OUTPUT_DIR + "/slm_replace_instrument.mid")

#%%

mask = model.tokenizer.replace_mask(x, ["instrument"]).to(model.device).float()

mask2 = (
    model.tokenizer.constraint_mask(
        instruments=["Organ","Piano"],
        min_notes=0,
    )[None, ...]
    .to(model.device)
    .float()
)

mask = mask * mask2

y = (
    model.generate(
        mask,
        temperature=1.0,
        schedule_fn=lambda x: x,
        top_p=1,
        top_k=0,
    )[0]
    .cpu()
    .numpy()
    .argmax(axis=-1)
)

y_sm = model.tokenizer.decode(y)

preview(y_sm, TMP_DIR)


y_sm.dump_midi(OUTPUT_DIR + "/slm_replace_instrument_w_constraint.mid")
#%%

mask = model.tokenizer.replace_mask(x, ["onset/beat","onset/tick","offset/beat","offset/tick"]).to(model.device).float()

y = (
    model.generate(
        mask,
        temperature=1.0,
        schedule_fn=lambda x: x,
        top_p=1,
        top_k=0,
    )[0]
    .cpu()
    .numpy()
    .argmax(axis=-1)
)

y_sm = model.tokenizer.decode(y)

preview(y_sm, TMP_DIR)

y_sm.dump_midi(OUTPUT_DIR + "/slm_replace_onset_offset.mid")

#%%

mask = (
    model.tokenizer.replace_mask(
        x, ["velocity"]
    )
    .to(model.device)
    .float()
)

y = (
    model.generate(
        mask,
        temperature=1.0,
        schedule_fn=lambda x: x,
        top_p=1,
        top_k=0,
    )[0]
    .cpu()
    .numpy()
    .argmax(axis=-1)
)

y_sm = model.tokenizer.decode(y)

preview(y_sm, TMP_DIR)

y_sm.dump_midi(OUTPUT_DIR + "/slm_replace_velocity.mid")


#%%

#%%

a = model.format_mask[None, ...].to(model.device)

# Generate a sequence
y = model.generate(
    a,
    schedule_fn=lambda x: x,
    temperature=0.99,
    top_p=1.0,
    top_k=0,
    fixed_order=False,
)[0].argmax(axis=1)

# decode
y_sm = model.tokenizer.decode(y)

print(f"Number of notes: {y_sm.note_num()}")

preview(y_sm, TMP_DIR)

y_sm.dump_midi(OUTPUT_DIR + "/no_constraint.mid")


#%%

x_sm = model.tokenizer.decode(x)

print(f"Number of notes: {x_sm.note_num()}")
# beat range
# beat_range=(8,12)
beat_range=(0,16)
pitch_range = [f"pitch:{i}" for i in range(55,108) ]+["pitch:-"]
# make infilling mask
mask = (
    model.tokenizer.infilling_mask(
        x,
        beat_range,
        min_notes=x_sm.note_num(),
        max_notes=x_sm.note_num(),
        pitches=pitch_range,
    )[None, ...]    
    .to(model.device)
    .float()
) 

plt.figure(figsize=(10,10))
plt.imshow(mask[0].cpu().numpy().T, aspect="auto",interpolation="nearest")
plt.show()


y = model.generate(
    mask,
    temperature=0.9,
    sampling_steps=300*9,
    top_p=1.0,
    top_k=0,
    order = "random"
)[0].cpu().numpy().argmax(axis=-1)
y_sm = model.tokenizer.decode(y)

# print n notes
print(f"Number of notes: {y_sm.note_num()}")

preview(y_sm, TMP_DIR)
y_sm.dump_midi(OUTPUT_DIR + "/infilling_high.mid")

#%%

x_sm = model.tokenizer.decode(x)

# beat range
# beat_range=(8,12)
beat_range = (0, 16)
pitch_range = [
    f"pitch:{i}" for i in range(model.tokenizer.config["pitch_range"][0], 55)
]
# make infilling mask
mask = (
    model.tokenizer.infilling_mask(
        x,
        beat_range,
        min_notes=x_sm.note_num(),
        max_notes=x_sm.note_num(),
        pitches=pitch_range,
    )[None, ...]
    .to(model.device)
    .float()
)

y = (
    model.generate(
        mask,
        temperature=0.85,
        top_p=1.0,
        top_k=0,
        order = "random"
    )[0]
    .cpu()
    .numpy()
    .argmax(axis=-1)
)
y_sm = model.tokenizer.decode(y)


preview(y_sm, TMP_DIR)
y_sm.dump_midi(OUTPUT_DIR + "/infilling_low.mid")


#%%


x_sm = model.tokenizer.decode(x)

beat_range = (4, 12)
pitch_range = [
    f"pitch:{i}" for i in range(40,60)
]

# make infilling mask
mask = (
    model.tokenizer.infilling_mask(
        x,
        beat_range=beat_range,
        min_notes=x_sm.note_num(),
        max_notes=x_sm.note_num(),
    )[None, ...]
    .to(model.device)
    .float()
)

y = (
    model.generate(
        mask,
        temperature=0.9,
        top_p=1.0,
        top_k=0,
        order = "random"
    )[0]
    .cpu()
    .numpy()
    .argmax(axis=-1)
)
y_sm = model.tokenizer.decode(y)


preview(y_sm, TMP_DIR)
y_sm.dump_midi(OUTPUT_DIR + "/infilling_box.mid")


#%%



#%%

# plot embedding self similarity

import matplotlib.pyplot as plt
import seaborn as sns

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
    sns.heatmap(embedding_similarity, cmap="magma", xticklabels=tokens, yticklabels=tokens, mask = np.eye(len(tokens)))
    plt.title(attr)
    plt.show()

    # plot the embedding_similarity matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(projection_similarity, cmap="magma", xticklabels=tokens, yticklabels=tokens, mask = np.eye(len(tokens)))
    plt.title(attr)
    plt.show()


#%%


# %%
