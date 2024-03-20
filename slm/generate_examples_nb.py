#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import piano_roll
import os
import IPython.display as ipd

device = "cuda:7"

ROOT_DIR = "../"

model = EncoderOnlyModel.load_from_checkpoint(
    ROOT_DIR+ "checkpoints/eager-darkness-234/epoch=65-step=76230-val/loss_epoch=0.15.ckpt",
    map_location=device,
)

MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR+"artefacts/tst_midi_records.pt",
    path_filter_fn=lambda x: f"n_bars={MODEL_BARS}" in x,
    genre_list=model.tokenizer.config["tags"],
    tokenizer=model.tokenizer,
    min_notes=8 * MODEL_BARS,
    max_notes=model.tokenizer.config["max_notes"],
)

OUTPUT_DIR = ROOT_DIR + "artefacts/examples"
TMP_DIR = ROOT_DIR + "artefacts/tmp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preview(sm, tmp_dir):
    # SAMPLE_RATE = 44_100
    os.makedirs(tmp_dir, exist_ok=True)
    midi_path = tmp_dir + "/tmp.mid"
    audio_path = tmp_dir + "/output.wav"
    sm.dump_midi(midi_path)
    pr = piano_roll(sm)
    plt.figure(figsize=(10, 10))
    sns.heatmap(pr, cmap="magma")
    plt.show()

    os.system(f"fluidsynth {midi_path} -F {audio_path}")
    ipd.display(ipd.Audio(audio_path))

#%%
    
# import random

# for i in random.sample(range(len(ds)), 3):
#     print(f"Example {i}")

#     x = ds[i]

#     # plot the piano roll
#     x_sm = model.tokenizer.decode(x)

#     preview(x_sm, TMP_DIR)

#%%

RESAMPLE_IDX = 17009

x = ds[RESAMPLE_IDX]
x_sm = model.tokenizer.decode(x)

x_sm.dump_midi(OUTPUT_DIR + "/resample_original.mid")

#%%

mask = model.tokenizer.replace_mask(x, ["pitch"]).to(model.device).float()

y = model.generate(
    mask,
    temperature=0.85,
)[0].cpu().numpy().argmax(axis=-1)

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

y_sm.dump_midi(OUTPUT_DIR + "/slm_replace_pitch_pentatonic.mid")


#%%

mask = model.tokenizer.replace_mask(x, ["instrument"]).to(model.device).float()

y = (
    model.generate(
        mask,
        temperature=0.9,
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
        instruments=["Bass","Guitar","Drums"],
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
        temperature=0.9,
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

a = model.format_mask[None, ...].to(model.device)
c = model.tokenizer.constraint_mask(
    # tags=["dance-eletric"],
    # tags=["other"],
    # instruments=["Ensemble"],
    # tempos = ["126"],
    scale="D pentatonic",
    min_notes=50,
    max_notes=250,
)[None, ...].to(model.device)
a = c * a

# Generate a sequence
y = model.generate(
    a
    schedule_fn=lambda x: x,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
)[0].argmax(axis=1)

# decode
y_sm = model.tokenizer.decode(y)

print(f"Number of notes: {y_sm.note_num()}")

preview(y_sm, TMP_DIR)

y_sm.dump_midi(OUTPUT_DIR + "/pitch_set_constraint_c.mid")


#%%

x_sm = model.tokenizer.decode(x)

print(f"Number of notes: {x_sm.note_num()}")
# beat range
# beat_range=(8,12)
beat_range=(0,16)
pitch_range = [f"pitch:{i}" for i in range(50,model.tokenizer.config["pitch_range"][1])]
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

y = model.generate(
    mask,
    temperature=1.0,
    sampling_steps=300*9,
    top_p=0.98
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
    f"pitch:{i}" for i in range(model.tokenizer.config["pitch_range"][0], 51)
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
        temperature=1.0,
        schedule_fn=lambda x: x,
        top_p=0.98,
        top_k=0,
    )[0]
    .cpu()
    .numpy()
    .argmax(axis=-1)
)
y_sm = model.tokenizer.decode(y)


preview(y_sm, TMP_DIR)
y_sm.dump_midi(OUTPUT_DIR + "//infilling_low.mid")


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
        temperature=1.0,
        schedule_fn=lambda x: x,
        top_p=0.95,
        top_k=0,
    )[0]
    .cpu()
    .numpy()
    .argmax(axis=-1)
)
y_sm = model.tokenizer.decode(y)


preview(y_sm, TMP_DIR)
y_sm.dump_midi(OUTPUT_DIR + "//infilling_middle.mid")


#%%

x_sm = model.tokenizer.decode(x)

# beat range
# beat_range=(8,12)
beat_range = (4, 12)

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
        temperature=1.0,
        schedule_fn=lambda x: x,
        top_p=0.98,
        top_k=0,
    )[0]
    .cpu()
    .numpy()
    .argmax(axis=-1)
)
y_sm = model.tokenizer.decode(y)


preview(y_sm, TMP_DIR)
y_sm.dump_midi(OUTPUT_DIR + "//infilling_box.mid")

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
