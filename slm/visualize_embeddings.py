#%%
# from train import EncoderOnlyModel
import numpy as np
import matplotlib.pyplot as plt
from train import TrainingWrapper

# model = SuperposedLanguageModel.load_from_checkpoint(
#     # "../checkpoints/faithful-wave-417/last.ckpt",
#     # "../checkpoints/smart-wood-419/last.ckpt",
#     # "../checkpoints/zesty-dawn-376/last.ckpt",
#     # "../checkpoints/cerulean-dragon-425/last.ckpt",
#     # "../checkpoints/unique-tree-426/last.ckpt",
#     # "../checkpoints/bumbling-dream-427/last.ckpt",
#     # "../checkpoints/lively-flower-428/last.ckpt",
#     # "../checkpoints/sparkling-dust-435/last.ckpt",
#     # "../checkpoints/misunderstood-eon-449/last.ckpt",
#     # "../checkpoints/chocolate-river-450/last.ckpt",
#     # "../checkpoints/fragrant-dew-452/last.ckpt",
#     # "../checkpoints/lilac-feather-455/last.ckpt",
#     # "../checkpoints/copper-monkey-456/last.ckpt",
#     # "../checkpoints/ruby-glade-461/last.ckpt",
#     # "../checkpoints/drawn-universe-463/last.ckpt",
#     # "../checkpoints/dulcet-jazz-464/last.ckpt",
#     "../checkpoints/stoic-capybara-480/last.ckpt",
#     map_location="cpu",
# )

# print(model.tokenizer.note_attribute_order)

# ckpt = paper_checkpoints.checkpoints["slm"]
# model = EncoderOnlyModel.load_from_checkpoint(
#     "../"+ckpt,
#     map_location="cpu",
# )

CHECKPOINTS = {
    "slm": "../checkpoints/usual-fire-530/last.ckpt",
    "mlm": "../checkpoints/toasty-bush-529/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "../checkpoints/balmy-deluge-532/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "../checkpoints/pretty-armadillo-542/last.ckpt",
    "slm_wo_enforce_constraint_in_fwd": "../checkpoints/colorful-sun-548/last.ckpt",
    "slm_not_norm_first": "../checkpoints/rural-oath-549/last.ckpt",
}

model = TrainingWrapper.load_from_checkpoint(
    # CHECKPOINTS["slm_not_norm_first"],
    CHECKPOINTS["slm"],
    map_location="cpu",
)

# %%

# set cmap to spectral everywhere
plt.rcParams["image.cmap"] = "Spectral"

vocab = model.tokenizer.vocab

embedding = model.model.embedding_layer.weight.detach().numpy().T

# imshow embeddings
plt.figure()
plt.imshow(embedding, interpolation="none")
plt.colorbar()
plt.show()


print(embedding.shape)

unembedding = model.model.unembedding_layer.weight.detach().numpy()

print(unembedding.shape)

# plot norms of embeddings, use log scale
plt.figure()
plt.plot(np.linalg.norm(embedding, axis=1))
plt.show()

plt.figure()
plt.plot(np.linalg.norm(unembedding, axis=1))
plt.show()

#%%


# normalize embeddings
embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
unembedding /= np.linalg.norm(unembedding, axis=1, keepdims=True)

print(embedding.shape, unembedding.shape)

# print self similarity

# take vocab and split it by ":"
vocab = [v.split(":")[0] for v in vocab]
# take only the first instance of each prefix, set others to " "
seen_prefixes = set()
for i, v in enumerate(vocab):
    prefix = v.split(" ")[0]
    if prefix in seen_prefixes:
        vocab[i] = " "
    else:
        seen_prefixes.add(prefix)

plt.rcParams["font.size"] = 10
plt.figure(figsize=(40, 40))
plt.imshow(embedding @ embedding.T, interpolation="none", vmin=-1, vmax=1)
# use vocab as yticks
plt.yticks(range(len(vocab)), vocab)
plt.colorbar()
plt.show()

plt.figure(figsize=(40, 40))
plt.imshow(unembedding @ unembedding.T, interpolation="none", vmin=-1, vmax=1)
# use vocab as yticks
plt.yticks(range(len(vocab)), vocab)
plt.colorbar()
plt.show()

# now take simlarity between embeddings and unembeddings
plt.figure(figsize=(40, 40))
plt.imshow(embedding @ unembedding.T, interpolation="none", vmin=-1, vmax=1)
# use vocab as yticks
plt.yticks(range(len(vocab)), vocab)
plt.colorbar()
plt.show()


#%%

# now only plot the embeddings for tokens that start with "onset/global_tick"

vocab = model.tokenizer.vocab
print(vocab)
onset_vocab = [v for v in vocab if v.startswith("onset/global_tick")]

# take first 25
onset_vocab = onset_vocab[:50]
onset_embeddings = embedding[[i for i, v in enumerate(vocab) if v in onset_vocab]] 


print(onset_embeddings.shape)
plt.figure(figsize=(20, 20))
# set larger font size
plt.rcParams["font.size"] = 18
plt.imshow(onset_embeddings @ onset_embeddings.T, interpolation="none", vmin=-1, vmax=1)
# use vocab as yticks


plt.yticks(range(len(onset_vocab)), onset_vocab)
# set x ticks to 90 degree rotation
plt.xticks(rotation=90)
plt.xticks(range(len(onset_vocab)), onset_vocab)
plt.colorbar()
plt.show()


#%% now show pitch embeddings
vocab = model.tokenizer.vocab
print(vocab)
pitch_vocab = [v for v in vocab if v.startswith("pitch")]

pitch_embeddings = embedding[[i for i, v in enumerate(vocab) if v in pitch_vocab]]

print(pitch_embeddings.shape)
plt.figure(figsize=(40, 40))
# set larger font size
plt.rcParams["font.size"] = 18
plt.imshow(pitch_embeddings @ pitch_embeddings.T, interpolation="none", vmin=-1, vmax=1)
# use vocab as yticks
plt.yticks(range(len(pitch_vocab)), pitch_vocab)
# set x ticks to 90 degree rotation
plt.xticks(rotation=90)
plt.xticks(range(len(pitch_vocab)), pitch_vocab)
plt.colorbar()
plt.show()


# %%

vocab = model.tokenizer.vocab
print(vocab)
duration_vocab = [v for v in vocab if v.startswith("duration")]

# take first 25
duration_vocab = duration_vocab[:50]
duration_embeddings = embedding[[i for i, v in enumerate(vocab) if v in duration_vocab]] 


print(duration_embeddings.shape)
plt.figure(figsize=(20, 20))
# set larger font size
plt.rcParams["font.size"] = 18
plt.imshow(duration_embeddings @ duration_embeddings.T, interpolation="none", vmin=-1, vmax=1)
# use vocab as yticks


plt.yticks(range(len(duration_vocab)), duration_vocab)
# set x ticks to 90 degree rotation
plt.xticks(rotation=90)
plt.xticks(range(len(duration_vocab)), duration_vocab)
plt.colorbar()
plt.show()

# %%

# plot velocity embeddings
vocab = model.tokenizer.vocab
velocity_vocab = [v for v in vocab if v.startswith("velocity")]

# take first 25

velocity_embeddings = embedding[[i for i, v in enumerate(vocab) if v in velocity_vocab]]

plt.figure(figsize=(40, 40))
# set larger font size
plt.rcParams["font.size"] = 18
plt.imshow(velocity_embeddings @ velocity_embeddings.T, interpolation="none", vmin=-1, vmax=1)
# use vocab as yticks
plt.yticks(range(len(velocity_vocab)), velocity_vocab)

#%%
# plot tempo embeddings
vocab = model.tokenizer.vocab
print(vocab)
tempo_vocab = [v for v in vocab if v.startswith("tempo")]
tempo_embeddings = embedding[[i for i, v in enumerate(vocab) if v in tempo_vocab]]
plt.figure(figsize=(40, 40))
plt.rcParams["font.size"] = 18
plt.imshow(tempo_embeddings @ tempo_embeddings.T, interpolation="none", vmin=-1, vmax=1)
plt.yticks(range(len(tempo_vocab)), tempo_vocab)
plt.colorbar()
plt.show()

#%% plot tag embeddings

vocab = model.tokenizer.vocab
print(vocab)

tag_vocab = [v for v in vocab if v.startswith("tag")]

tag_embeddings = embedding[[i for i, v in enumerate(vocab) if v in tag_vocab]]

plt.figure(figsize=(40, 40))
plt.rcParams["font.size"] = 18

plt.imshow(tag_embeddings @ tag_embeddings.T, interpolation="none", vmin=-1, vmax=1)
plt.yticks(range(len(tag_vocab)), tag_vocab)
plt.colorbar()

plt.show()




# %%


# apply tsne to pitch embeddings

from sklearn.manifold import TSNE
# use PCA
from sklearn.decomposition import PCA
#reducer = TSNE(n_components=2, perplexity=30, n_iter=1000)
reducer = PCA(n_components=2)
reducer_3d = PCA(n_components=3)

#%%

pitch_vocab = [v for v in vocab if ("pitch" in v) and "Drums" not in v]

pitch_embeddings = embedding[[i for i, v in enumerate(vocab) if v in pitch_vocab]]


pitch_reduced = reducer.fit_transform(pitch_embeddings)

pitch_values = [v.split(":")[-1] for v in pitch_vocab]

# plot with pitch names
plt.figure()
plt.scatter(pitch_reduced[:, 1], pitch_reduced[:, 0], c=range(len(pitch_vocab)))

for i, txt in enumerate(pitch_vocab):
    plt.annotate(txt.split(":")[-1], (pitch_reduced[i, 1], pitch_reduced[i, 0]),size=6)

#%%
# now plot in 3d

pitch_reduced_3d = reducer_3d.fit_transform(pitch_embeddings)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pitch_reduced_3d[:, 0], pitch_reduced_3d[:, 1], pitch_reduced_3d[:, 2], c=range(len(pitch_vocab)))

for i, txt in enumerate(pitch_vocab):
    ax.text(pitch_reduced_3d[i, 0], pitch_reduced_3d[i, 1], pitch_reduced_3d[i, 2], txt.split(":")[-1],size=6)

plt.show()

#%%

# same for onset

onset_vocab = [v for v in vocab if ("onset" in v) and "Drums" not in v]

onset_embeddings = embedding[[i for i, v in enumerate(vocab) if v in onset_vocab]]

onset_reduced = reducer.fit_transform(onset_embeddings)

onset_values = [v.split(":")[-1] for v in onset_vocab]

plt.figure()
plt.scatter(onset_reduced[:, 1], onset_reduced[:, 0], c=range(len(onset_vocab)))

for i, txt in enumerate(onset_vocab):
    plt.annotate(txt.split(":")[-1], (onset_reduced[i, 1], onset_reduced[i, 0]), size=6)


# %%

# do same for tags

tag_vocab = [v for v in vocab if ("tag" in v) and "Drums" not in v]

tag_embeddings = embedding[[i for i, v in enumerate(vocab) if v in tag_vocab]]

tag_reduced = reducer.fit_transform(tag_embeddings)

tag_values = [v.split(":")[-1] for v in tag_vocab]

plt.figure()

plt.scatter(tag_reduced[:, 1], tag_reduced[:, 0], c=range(len(tag_vocab)))

for i, txt in enumerate(tag_vocab):
    plt.annotate(txt.split(":")[-1], (tag_reduced[i, 1], tag_reduced[i, 0]))

# %%
