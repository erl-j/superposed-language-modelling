#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dense_train import UnetModel
from dense_transformer_train import DenseModel
from util import piano_roll

#%%

device = "cuda:7"


model = DenseModel.load_from_checkpoint(
    "../checkpoints/stellar-wood-46/epoch=10-step=73910-val/loss=0.03-trn/loss=0.05.ckpt",
    map_location=device,
)

# model = DenseModel.load_from_checkpoint(
#     "../checkpoints/expert-grass-60/epoch=3-step=50775-trn/loss=0.00.ckpt",
#     map_location=device,
# )
#%%
# Move the model to the device
model = model.to(device)
# Generate a sequence
a = a = model.tokenizer.get_format_mask()[None, ...].to(model.device)
#%%
# get logits
logits = model.forward(a.float())

probs = torch.nn.functional.softmax(logits, dim=-1)
# get probability of holds
holds = probs[0, ..., 0, model.tokenizer.vocab2idx["action:hold"]]

plt.imshow(holds.cpu().detach().numpy().T, aspect="auto", interpolation="none")
plt.show()

# get probailities of onsets
onset_idxs = [model.tokenizer.vocab2idx[f"action vel:{i}"] for i in range(1, 128)]

onsets = probs[0, ..., 0, onset_idxs].sum(dim=-1)

plt.imshow(onsets.cpu().detach().numpy().T, aspect="auto", interpolation="none")
plt.show()
#%%

import symusic
s = model.generate(a, sampling_steps=500, temperature=0.94)


def merge_tracks(sm):
    all_notes = []
    new_tracks = []
    for track in sm.tracks:
        if track.is_drum:
            # add the drum track
            new_tracks.append(track)
        else:
            for note in track.notes:
                all_notes.append(note)
    
    # sort notes by start time
    all_notes.sort(key=lambda x: x.start)
    # create a new track
    new_track = symusic.Track(program=0, is_drum=False, name="merged")
    # add notes to the new track
    for note in all_notes:
        new_track.notes.append(note)
    # remove all tracks
    # add the new track
    sm.tracks = new_tracks + [new_track]
    return sm

# dump

# argmax
s_idx = s.cpu().numpy().argmax(axis=-1)[0]

sm = model.tokenizer.decode(s_idx)

sm = merge_tracks(sm)
# save midi
sm.dump_midi("../artefacts/generated_dense.mid")

pr = piano_roll(sm)

plt.imshow(pr, aspect="auto")
plt.show()
#%%

# get the embedding
# embedding = model.embedding_layer.weight.detach().cpu().numpy()

embedding = model.time_embedding.weight.detach().cpu().numpy().T

embedding = model.voice_embedding.weight.detach().cpu().numpy().T
# get vocab
# vocab = model.tokenizer.vocab

norm = np.linalg.norm(embedding, axis=0)

# compute the similarity matrix
similarity = (embedding.T @ embedding) / (norm @ norm.T)

# plot the similarity matrix
# make large figure
plt.figure(figsize=(30, 30))
# set small font
sns.set(font_scale=0.5)
sns.heatmap(similarity, cmap="magma")
plt.show()

#%%

logits = model.forward(a.float())

print(logits.shape)


#%%

