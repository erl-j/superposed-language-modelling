#%%
from train2 import SuperposedLanguageModel

import numpy as np
#%%

model = SuperposedLanguageModel.load_from_checkpoint(
    "../checkpoints/snowy-universe-375/epoch=9-step=64850-val/loss_epoch=0.22075.ckpt",
    map_location="cpu",
)
# %%

vocab = model.vocab

embedding = model.embedding_layer.weight.detach().numpy().T

unembedding = model.decoder_output_layer.weight.detach().numpy()

# normalize embeddings
embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
unembedding /= np.linalg.norm(unembedding, axis=1, keepdims=True)

print(embedding.shape, unembedding.shape)

# print self similarity
import matplotlib.pyplot as plt

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

# %%
