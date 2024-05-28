#%%
#from h_causal import HierarchicalCausalDecoderModel
from h_causal_w_prior import HierarchicalCausalDecoderModel

device = "cuda:0"
# Load the model
model = HierarchicalCausalDecoderModel.load_from_checkpoint(
    # "../checkpoints/glowing-lion-18/last.ckpt",
    "../checkpoints/copper-cosmos-4/last.ckpt",
    #"../checkpoints/celestial-microwave-7/last.ckpt",
    # "../checkpoints/still-microwave-5/last.ckpt",
    map_location=device,
)

mask = model.tokenizer.constraint_mask(
    tags = ["pop"],
    tempos = ["128"],
    instruments =["Piano"],
    scale = "G major",
    min_notes=10,
    max_notes=290,
    min_notes_per_instrument=50
)[None,:]

# mask = model.tokenizer.get_format_mask()[None,...]
x = model.sample(mask,temperature=0.99, force_mask=True)

#%%
print(model.tokenizer.indices_to_tokens(x.flatten()))

#%%

from util import preview_sm
x_sm = model.tokenizer.decode(x.flatten())
preview_sm(x_sm)
print(x_sm.note_num())
# %%
print(x.shape)
print(model.tokenizer.indices_to_tokens(x.flatten()))
# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# get the embedding
embedding = model.embedding_layer.weight.detach().cpu().numpy()
prior_embedding = model.prior_embedding_layer.weight.detach().cpu().numpy().T
projection = model.decoder_output_layer.weight.detach().cpu().numpy()

print(embedding.shape)
print(projection.shape)
print(prior_embedding.shape)

# get vocab
vocab = model.tokenizer.vocab

embedding_norm = np.linalg.norm(embedding, axis=0, keepdims=True)
prior_embedding_norm = np.linalg.norm(prior_embedding, axis=0, keepdims=True)
projection_norm = np.linalg.norm(projection, axis=0, keepdims=True)

# normalize the embedding
embedding = embedding / embedding_norm
projection = projection / projection_norm
prior_embedding = prior_embedding / prior_embedding_norm

# compute the cosine similarity
embedding_similarity = embedding @ embedding.T
projection_similarity = projection @ projection.T
prior_embedding_similarity = prior_embedding @ prior_embedding.T


cmap = "Spectral_r"
# plot the similarity matrix
# make large figure
plt.figure(figsize=(50, 50))
# set small font
sns.set(font_scale=0.5)
sns.heatmap(
    embedding_similarity,
    cmap=cmap,
    xticklabels=vocab,
    yticklabels=vocab,
    mask=np.eye(len(vocab)),
    vmax=1,
    vmin=-1,
)
plt.show()

plt.figure(figsize=(50, 50))
# set small font
sns.set(font_scale=0.5)
sns.heatmap(
    prior_embedding_similarity,
    cmap=cmap,
    xticklabels=vocab,
    yticklabels=vocab,
    mask=np.eye(len(vocab)),
    vmax=1,
    vmin=-1,
)
plt.show()

# plot the similarity matrix
# make large figure
plt.figure(figsize=(50, 50))
# set small font
sns.set(font_scale=0.5)
sns.heatmap(
    projection_similarity,
    cmap=cmap,
    xticklabels=vocab,
    yticklabels=vocab,
    mask=np.eye(len(vocab)),
    vmax=1,
    vmin=-1,
)
plt.show()
# %%
