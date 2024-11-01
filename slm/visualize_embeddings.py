#%%
from train2 import SuperposedLanguageModel
from train import EncoderOnlyModel
import numpy as np
import matplotlib.pyplot as plt
#%%
import paper_checkpoints

model = SuperposedLanguageModel.load_from_checkpoint(
    "../checkpoints/luminous-marigold-398/last.ckpt",
    # "../checkpoints/resplendent-wick-397/last.ckpt",
    # "../checkpoints/chromatic-triumph-396/last.ckpt",
    # "../checkpoints/zesty-dawn-376/last.ckpt",
    # "../checkpoints/ghostly-pulse-378/last.ckpt",
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

embedding = model.embedding_layer.weight.detach().numpy()

print(model.embedding_layer.all_atoms)

print(embedding.shape)

unembedding = model.decoder_output_layer.weight.detach().numpy()

# plot norms of embeddings, use log scale
plt.figure()
plt.plot(np.log(np.linalg.norm(embedding, axis=1)))
plt.show()

plt.figure()
plt.plot(np.log(np.linalg.norm(unembedding, axis=1)))
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
plt.imshow(embedding @ embedding.T, interpolation="none")
# use vocab as yticks
plt.yticks(range(len(vocab)), vocab)
plt.colorbar()
plt.show()

plt.figure(figsize=(40, 40))
plt.imshow(unembedding @ unembedding.T, interpolation="none")
# use vocab as yticks
plt.yticks(range(len(vocab)), vocab)
plt.colorbar()
plt.show()

# now take simlarity between embeddings and unembeddings
plt.figure(figsize=(40, 40))
plt.imshow(embedding @ unembedding.T, interpolation="none")
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
onset_vocab = onset_vocab[:55]
onset_embeddings = embedding[[i for i, v in enumerate(vocab) if v in onset_vocab]] 


print(onset_embeddings.shape)
plt.figure(figsize=(20, 20))
# set larger font size
plt.rcParams["font.size"] = 18
plt.imshow(onset_embeddings @ onset_embeddings.T, interpolation="none")
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
plt.figure(figsize=(20, 20))
# set larger font size
plt.rcParams["font.size"] = 18
plt.imshow(pitch_embeddings @ pitch_embeddings.T, interpolation="none")
# use vocab as yticks
plt.yticks(range(len(pitch_vocab)), pitch_vocab)
# set x ticks to 90 degree rotation
plt.xticks(rotation=90)
plt.xticks(range(len(pitch_vocab)), pitch_vocab)
plt.colorbar()
plt.show()


# %%
