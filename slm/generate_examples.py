#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from merged_encoder_only_train import EncoderOnlyModel
from util import piano_roll
from data import MidiDataset
import os

#%%

device = "cuda:7"

model = EncoderOnlyModel.load_from_checkpoint(
    "../checkpoints/eager-darkness-234/epoch=65-step=76230-val/loss_epoch=0.15.ckpt",
    map_location=device,
)
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
    cache_path="./artefacts/val_midi_records.pt",
    path_filter_fn=lambda x: f"n_bars={N_BARS}" in x,
    genre_list=genre_list,
    tokenizer=model.tokenizer,
    min_notes=8 * N_BARS,
    max_notes=model.tokenizer.config["max_notes"],
)

#%%
OUTPUT_DIR = "./artefacts/examples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

x = val_ds[20]
x_sm = model.tokenizer.decode(x)
pr = piano_roll(x_sm)
# dump original
x_sm.dump_midi(os.path.join(OUTPUT_DIR, "original.mid"))

# Resample pitches


# Resample onsets / offsets

# Resample velocities

# Resample drums


#%%


# plot the piano roll


print(f"Number of notes: {x_sm.note_num()}")
      
# beat range
# beat_range=(8,12)
beat_range=(0,16)
pitch_range = [f"pitch:{i}" for i in range(50,model.tokenizer.config["pitch_range"][1])]

# make infilling mask
mask = model.tokenizer.infilling_mask(x,
                                      beat_range,
                                    #   max_notes = 270,
                                     max_notes=x_sm.note_num(),
                                    pitches=pitch_range,
                                      )[None,...].to(model.device).float()

y = model.generate(
    mask,
    temperature=1.0,
    sampling_steps=300*9,
    schedule_fn=lambda x: x,
    top_p=1,
    top_k=20,
)

y_idx = y[0].cpu().numpy().argmax(axis=1)

y_sm = model.tokenizer.decode(y_idx)

print(f"Number of notes: {y_sm.note_num()}")

pr2 = piano_roll(y_sm)
# use a grid


x_sm.dump_midi("../artefacts/infill_original.mid")

plt.figure(figsize=(10, 10))
sns.heatmap(pr, cmap="magma")
plt.vlines(beat_range[0] * 4, 0, pr.shape[0], color="white")
plt.vlines(beat_range[1] * 4, 0, pr.shape[0], color="white")
plt.show()

# add h lines for the beat range
plt.figure(figsize=(10, 10))
sns.heatmap(pr2, cmap="magma")
plt.vlines(beat_range[0] * 4, 0, pr2.shape[0], color="white")
plt.vlines(beat_range[1] * 4, 0, pr2.shape[0], color="white")
plt.show()

# save
y_sm.dump_midi("../artefacts/infill_result.mid")

#%%

plt.figure(figsize=(10, 10))

x = val_ds[30]

mask = model.tokenizer.shuffle_notes_mask(x)[None, ...].to(model.device).float()

y = model.generate(
    mask,
    sampling_steps=300*9,
    schedule_fn=lambda x: x,
    temperature=1,
    # top_p=1,
    top_k=0,
)

x_sm = model.tokenizer.decode(x)

# print tempo
print(x_sm.tempos[0])

# print number of notes
print(x_sm.note_num())

pr = piano_roll(x_sm)

x_sm.dump_midi("../artefacts/shuffle_original.mid")

y_sm = model.tokenizer.decode(y[0].cpu().numpy().argmax(axis=1))

pr2 = piano_roll(y_sm)

# print number of notes
print(y_sm.note_num())

plt.figure(figsize=(10, 10))
sns.heatmap(pr, cmap="magma")
plt.show()

# add h lines for the beat range
plt.figure(figsize=(10, 10))
sns.heatmap(pr2, cmap="magma")
plt.show()

# save
y_sm.dump_midi("../artefacts/shuffle_result.mid")


#%%


#%%
# Generate a sequence
a = model.format_mask[None,...].to(model.device)

# print(model.tokenizer.tempo_bins)

c = model.tokenizer.constraint_mask(
    # tags=["dance-eletric"],
    # tags=["other"],
    # instruments=["Ensemble"],
    # tempos = ["126"],
    scale = "G major",
    min_notes=120,
    max_notes=180,
)[None,...].to(model.device)
a = c*a


# Generate a sequence
sequence = model.generate(a,
                        sampling_steps=300*9,
                        schedule_fn=lambda x: x, 
                        temperature=0.9,
                        top_p = 1.0,
                        # top_k=30,
                        )


token_idx = sequence[0].argmax(axis=1)

# decode
sm = model.tokenizer.decode(token_idx)

# print number of notes
print(sm.note_num())

pr = piano_roll(sm)

plt.figure(figsize=(10, 10))
sns.heatmap(pr, cmap="magma")
plt.show()

# save the sequence
sm.dump_midi("../artefacts/generated.mid")

#%%

# plot embedding self similarity

import matplotlib.pyplot as plt
import seaborn as sns

# get the embedding
embedding = model.embedding_layer.weight.detach().cpu().numpy()

# get vocab
vocab = model.tokenizer.vocab

# compute the similarity matrix
similarity = embedding.T @ embedding

# plot the similarity matrix
# make large figure
plt.figure(figsize=(10, 10))
# set small font
sns.set(font_scale=0.5)
sns.heatmap(similarity, cmap="magma", xticklabels=vocab, yticklabels=vocab)
plt.show()

#%%


# %%
