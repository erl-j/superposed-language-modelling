# %% Imports
import torch
import numpy as np
from train import TrainingWrapper
from data import MidiDataset
from slm.PAPER_CHECKPOINTS import CHECKPOINTS
from tqdm import tqdm

# set seed
torch.manual_seed(0)
# %% Configure settings
MODEL_NAME = "slm_sparse_150epochs"  # Replace with your model name
N_BATCHES = 100
BATCH_SIZE = 60
GPU = 0
device = f"cuda:{GPU}" if torch.cuda.is_available() else "cpu"

# %% Load model
model = TrainingWrapper.load_from_checkpoint(CHECKPOINTS[MODEL_NAME], map_location=device)
model.eval()

# %% Setup dataset
mmd_4bar_filter_fn = lambda x: "n_bars=4" in x
sm_filter_fn = lambda sm: not any(
    track.program == 0 and not track.is_drum and "piano" not in track.name.lower()
    for track in sm.tracks
)

val_ds = MidiDataset(
    cache_path="../data/mmd_loops/val_midi_records_unique_pr.pt",
    path_filter_fn=mmd_4bar_filter_fn,
    genre_list=model.tokenizer.config["tags"],
    tokenizer=model.tokenizer,
    min_notes=16,
    max_notes=model.tokenizer.config["max_notes"],
    use_random_shift=False,
    sm_filter_fn=sm_filter_fn,
)

val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

# %% Collect activations
all_activations = []
records = []
with torch.no_grad():
    for i in tqdm(range(N_BATCHES)):
        batch = next(iter(val_dl))
        token_ids = batch["token_ids"].to(device)
        for j in range(token_ids.size(0)):
            # flatten first
            decoded = model.tokenizer.indices_to_tokens(token_ids[j].flatten().cpu().numpy())
            # get first non "tag:..."" token that is not "tag:-""
            for k, token in enumerate(decoded):
                if token.startswith("tag:") and token != "tag:-":
                    tag = token
                    break
            # get tempo token
            for k, token in enumerate(decoded):
                if token.startswith("tempo:") and token != "tempo:-":
                    tempo = token
                    break
            records.append({"tag": tag, "tempo": tempo})
        # convert to one-hot
        activations = model.model.embed(token_ids, mask_attributes = ("tag", "tempo")).detach()
        all_activations.append(activations.cpu().numpy())


#%% save activations
activations = np.concatenate(all_activations, axis=0)
# shape is (samples, events, features)
#%%
# take mean across events
mean_activations = np.mean(activations, axis=1)

#%%
# plot means
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.figure(figsize=(12, 8))
sns.heatmap(mean_activations, cmap="viridis")
plt.title("Mean activations")
plt.show()

#%%
# take pca of activations
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


proj = TSNE(n_components=2, perplexity=30, n_iter=300)
#proj = PCA(n_components=2)

proj_activations = proj.fit_transform(mean_activations)

#%% plot scatter
# colour code by tag
tags = [record["tag"] for record in records]
unique_tags = list(set(tags))

plt.figure(figsize=(12, 8))
for i, tag in enumerate(unique_tags):
    idx = np.where(np.array(tags) == tag)[0]
    # hide other tags
    if tag != "tag:other":
        plt.scatter(proj_activations[idx, 0], proj_activations[idx, 1], label=tag)

plt.legend()
plt.title("TSNE of activations")
plt.show()

#%% 
# colour code by tempo

tempos = [record["tempo"] for record in records]
unique_tempos = list(set(tempos))

tempo_range = (40, 300)

# make colour map to map tempos to colours
cmap = plt.get_cmap("viridis")

def tempo_token_to_float(token):
    return int(token.split(":")[1])

tempos = np.array([tempo_token_to_float(tempo) for tempo in tempos])

plt.figure(figsize=(12, 8))

plt.scatter(proj_activations[:, 0], proj_activations[:, 1], c=tempos, cmap=cmap, vmin=tempo_range[0], vmax=tempo_range[1])
plt.legend()
plt.title("TSNE of activations")
plt.show()


# %%
