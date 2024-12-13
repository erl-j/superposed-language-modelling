#%%
from train2 import SuperposedLanguageModel
# from train import EncoderOnlyModel
import numpy as np
import matplotlib.pyplot as plt
#%%
import paper_checkpoints

model = SuperposedLanguageModel.load_from_checkpoint(
    # "../checkpoints/faithful-wave-417/last.ckpt",
    # "../checkpoints/smart-wood-419/last.ckpt",
    # "../checkpoints/zesty-dawn-376/last.ckpt",
    # "../checkpoints/cerulean-dragon-425/last.ckpt",
    # "../checkpoints/unique-tree-426/last.ckpt",
    # "../checkpoints/bumbling-dream-427/last.ckpt",
    # "../checkpoints/lively-flower-428/last.ckpt",
    # "../checkpoints/sparkling-dust-435/last.ckpt",
    # "../checkpoints/misunderstood-eon-449/last.ckpt",
    # "../checkpoints/chocolate-river-450/last.ckpt",
    # "../checkpoints/fragrant-dew-452/last.ckpt",
    # "../checkpoints/lilac-feather-455/last.ckpt",
    # "../checkpoints/copper-monkey-456/last.ckpt",
    # "../checkpoints/ruby-glade-461/last.ckpt",
    # "../checkpoints/drawn-universe-463/last.ckpt",
    # "../checkpoints/dulcet-jazz-464/last.ckpt",
    "../checkpoints/stoic-capybara-480/last.ckpt",
    map_location="cpu",
)

print(model.tokenizer.note_attribute_order)

# ckpt = paper_checkpoints.checkpoints["slm"]
# model = EncoderOnlyModel.load_from_checkpoint(
#     "../"+ckpt,
#     map_location="cpu",
# )
# %%

# set cmap to spectral everywhere
plt.rcParams["image.cmap"] = "Spectral"

vocab = model.vocab

embedding = model.embedding_layer.weight.detach().numpy().T

# imshow embeddings
plt.figure()
plt.imshow(embedding, interpolation="none")
plt.colorbar()
plt.show()


print(embedding.shape)

unembedding = model.decoder_output_layer.weight.detach().numpy()

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

vocab = model.vocab
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
vocab = model.vocab
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

vocab = model.vocab
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
vocab = model.vocab
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
vocab = model.vocab
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

vocab = model.vocab
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
