#%%
from simpleflow import SimpleFlowModel
import torch
from data import MidiDataset
import copy
from util import preview_sm
device = 'cuda:4'

ckpt = "../checkpoints/light-bee-20/last.ckpt"
# ckpt = "../checkpoints/eager-vortex-26/last.ckpt"
# ckpt = "../checkpoints/hardy-bush-25/last.ckpt"
model = SimpleFlowModel.load_from_checkpoint(
    ckpt, map_location="cpu"
).to(device)

#%%

dataset = "mmd_loops"

ROOT_DIR = "../"
TMP_DIR = ROOT_DIR + "artefacts/tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/output"

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

RESAMPLE_IDX = 1500

x = ds[RESAMPLE_IDX]
x_sm = model.tokenizer.decode(x)

preview_sm(x_sm)

#%%

y = model.sample(n_steps= 100, temperature=1)

y_sm = model.tokenizer.decode(y[0].cpu().numpy())

print(y_sm.note_num())

preview_sm(y_sm)

#%%

mask = model.tokenizer.infilling_mask(
    x=x,
    beat_range=(4, 12),
    min_notes=x_sm.note_num(),
    max_notes=x_sm.note_num()
)

# mask = model.tokenizer.constraint_mask(
#     instruments=["Drums","Bass","Guitar"],
#     min_notes = 10,
#     max_notes=  100,
#     min_notes_per_instrument=10
# )

mask = torch.tensor(mask * model.tokenizer.get_format_mask()).float()

prior = mask / mask.sum(dim=-1, keepdim=True)[None,:]

l,y = model.sample(
    prior=prior,
    sampling_args=sampling_args,
    break_on_anomaly=True,
    log= True,
)

y = y.argmax(dim=-1)

y_sm = model.tokenizer.decode(y[0].cpu().numpy())

preview_sm(y_sm)

# %%

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

embedding = model.E.data.detach().cpu().numpy()
projection = model.U.data.detach().cpu().numpy().T

print(embedding.shape)
print(projection.shape)

# get vocab
vocab = model.tokenizer.vocab

embedding_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
projection_norm = np.linalg.norm(projection, axis=1, keepdims=True)
print(embedding_norm.shape)
print(projection_norm.shape)

# bar plot of embedding norms with vocab as y labels
# bars go left to right
plt.figure(figsize=(5, 40))
plt.plot( embedding_norm.flatten(),vocab, color="blue")
plt.xlabel("Embedding Norm")
plt.ylabel("Vocabulary")
plt.title("Embedding Norms")
plt.show()

plt.figure(figsize=(5, 40))
plt.plot( projection_norm.flatten(),vocab, color="blue")
plt.xlabel("Embedding Norm")
plt.ylabel("Vocabulary")
plt.title("Embedding Norms")
plt.show()





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

# do pca on embeddings

#%%
from sklearn.decomposition import PCA

attributes = [v.split(":")[0] for v in vocab]
values = [v.split(":")[1] for v in vocab]

# color according to label
# pca = PCA(n_components=2)
# embedding_pca = pca.fit_transform(embedding)

# plt.figure(figsize=(10, 10))
# sns.scatterplot(x=embedding_pca[:,0], y=embedding_pca[:,1], hue=attributes, palette="tab20")
# plt.title("PCA of Embedding")
# plt.show()

# fit mds model on similarities
# use tsne instead
from sklearn.manifold import MDS, TSNE


for zs in [embedding, projection]:
    # mds = MDS(n_components=2, dissimilarity="precomputed")
    m = TSNE(n_components=2, perplexity=30)
    zs_mds = m.fit_transform(zs)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=zs_mds[:,0], y=zs_mds[:,1], hue=attributes, palette="tab20")
    for i, txt in enumerate(values):
        plt.annotate(txt, (zs_mds[i,0], zs_mds[i,1]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title("TSNE of Embedding")
    plt.show()



# %%
