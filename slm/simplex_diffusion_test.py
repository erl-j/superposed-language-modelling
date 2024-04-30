#%%
device = "cuda:0"
from simplex_diffusion import SimplexDiffusionModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from util import preview_sm,piano_roll, load_merged_models, clear_overlap_notes, sm_reduce_dynamics
import matplotlib.pyplot as plt
import seaborn as sns
from data import MidiDataset
import torch
from tqdm import tqdm
import glob
import os

#%%

#model = SimplexDiffusionModel.load_from_checkpoint("../checkpoints/dark-sky-67/last.ckpt").to(device)
model = SimplexDiffusionModel.load_from_checkpoint("../checkpoints/flowing-paper-64/last.ckpt").to(device)
#model = load_merged_models("../checkpoints/flowing-paper-64/**/*.ckpt",SimplexDiffusionModel).to(device)

#model = SimplexDiffusionModel.load_from_checkpoint("../checkpoints/daily-armadillo-82/last.ckpt").to(device)
#model = load_merged_models("../checkpoints/daily-armadillo-82/**/*.ckpt",SimplexDiffusionModel).to(device)
#model = load_merged_models("../checkpoints/peachy-silence-79/**/*.ckpt",SimplexDiffusionModel).to(device)

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

print(len(ds))
#%%
# 1403
# 3050 guitar
# 3700?
RESAMPLE_IDX = 1400

x = ds[RESAMPLE_IDX]
x_sm = model.tokenizer.decode(x)

preview_sm(x_sm)

print(f"Number of notes: {x_sm.note_num()}")
#%%
x2 = ds[RESAMPLE_IDX+500]
x_sm2 = model.tokenizer.decode(x2)

preview_sm(x_sm2)
#%%

# %%
sns.set_style("whitegrid", {'axes.grid' : False})




mask = model.tokenizer.constraint_mask(
    # tags=["funk"],
    # tags=["chords"],
    # scale="C major",
    # tags=["alternative-indie"],

    instruments=["Piano"],
    tempos=["126"],
    min_notes = 40,
    max_notes = 150,
    # max_notes = 290,
    # min_notes_per_instrument=10,
)


mask = model.tokenizer.infilling_mask(
    x=x,
    beat_range=(4, 12),
    min_notes=0,
    max_notes=290,
)


beat_range=(0,16)
#pitch_range = [f"pitch:{i}" for i in range(50,108) ]+["pitch:-"]
pitch_range = [f"pitch:{i}" for i in range(0,50) ]+["pitch:-"]
#make infilling mask
mask = (
    model.tokenizer.infilling_mask(
        x,
        beat_range,
        min_notes=x_sm.note_num(),
        max_notes=x_sm.note_num(),
        pitches=pitch_range,
        mode ="harmonic"
    )[None, ...]    
    .to(model.device)
    .float()
) 

# chromatic percussion is fun!
#mask = model.tokenizer.replace_instruments(x,["Piano","Guitar"],["Chromatic Percussion"],20,275)[None, ...].to(model.device).float()
mask = model.tokenizer.replace_instruments(x,["Piano","Guitar"],["Piano","Guitar"],20,275)[None, ...].to(model.device).float()

# mask = torch.nn.functional.one_hot(x, num_classes=len(model.tokenizer.vocab)).float()



format_mask = torch.tensor(model.tokenizer.get_format_mask(), device=model.device).float()

mask = mask.to(model.device).float()


# plt.imshow(mask.cpu().numpy().T, aspect="auto",interpolation="none")
# plt.title("Mask")
# plt.show()

# plt.imshow(format_mask.cpu().numpy().T, aspect="auto",interpolation="none")
# plt.title("Format Mask")
# plt.show()

# plt.imshow((mask*format_mask).cpu().T, aspect="auto",interpolation="none")
# plt.title("Mask * Format Mask")
# plt.show()

# assert torch.allclose(mask, mask*format_mask)


# mask = torch.ones_like(format_mask)
# mask = mask * format_mask


# set torch seed
torch.manual_seed(1)
# infilling top-p 0.5
BATCH_SIZE = 3
N_STEPS = 50
TOP_P = 0.75
PRIOR_STRENGTH = 1.0
REFINEMENT_STEPS = 0
ENFORCE_MULTIPLY = True

prior = mask / mask.sum(dim=-1, keepdim=True)

plt.imshow(prior.cpu().numpy().T, aspect="auto",interpolation="none")
plt.title("Prior")
plt.show()
# check that the prior is normalized
assert torch.allclose(prior.sum(dim=-1), torch.ones_like(prior.sum(dim=-1)))
plt.imshow(prior.cpu().numpy().T, aspect="auto",interpolation="none")
plt.show()


y = model.sample2(prior,
                BATCH_SIZE,
                N_STEPS,
                top_p=TOP_P,
                prior_strength=PRIOR_STRENGTH,
                plot=False,
                enforce_prior=True,
                enforce_multiply=ENFORCE_MULTIPLY,
                decay_prior=False,
                inverse_decay=False,
                attribute_temperature=None,
                )

# # add refinement step
# # one hot
# if REFINEMENT_STEPS > 0:
#     ys = torch.nn.functional.one_hot(y, num_classes=len(model.tokenizer.vocab)).float()
#     # soften my mixing with format mask
#     format_prior = format_mask / format_mask.sum(dim=-1, keepdim=True)
#     alpha = 0.5
#     ys = alpha * ys + (1-alpha) * format_prior
#     # renormalize
#     ys = ys / ys.sum(dim=-1, keepdim=True)
#     # multiply with prior
#     ys = ys * prior
#     # renormalize
#     ys = ys / ys.sum(dim=-1, keepdim=True)


#     # refine
#     y2 = model.sample2(prior,
#                     BATCH_SIZE,
#                     REFINEMENT_STEPS,
#                     top_p=1.0,
#                     prior_strength=0.5,
#                     plot=False,
#                     enforce_prior=True,
#                     enforce_multiply=False,
#                     decay_prior=False,
#                     attribute_temperature=None,
#                     inverse_decay=False,
#                     )
# plot before and after
for i in range(BATCH_SIZE):
    y_sm = model.tokenizer.decode(y[i])
    print(f"Number of notes: {y_sm.note_num()}")
    y_sm = clear_overlap_notes(y_sm)
    print(f"Number of notes after clearing overlap: {y_sm.note_num()}")
    preview_sm(y_sm)
    # export to midi
    sm_reduce_dynamics(y_sm, 0.5)
    os.makedirs("../artefacts/test_vel",exist_ok=True)
    y_sm.dump_midi(f"../artefacts/test_vel/sample_{i}.mid")
    # if REFINEMENT_STEPS > 0:
    #     y2_sm = model.tokenizer.decode(y2[i])
    #     preview_sm(y2_sm)

    # print(f"Number of notes: {y2_sm.note_num()}")
    # print("\n\n")

    # ce = model.self_eval(y, None, None, t=1).detach().cpu()
    # print(ce.shape)
    # plt.plot(ce.detach().cpu())
    # plt.show()

    # # sort by the lowest cross entropy
    # idx = ce.argsort()
    # for i in range(5):
    #     y_sm = model.tokenizer.decode(y[idx[i]])
    #     print(f"Cross Entropy: {ce[idx[i]]}")
    #     print(f"Number of notes: {y_sm.note_num()}")
    #     preview_sm(y_sm)
    #     print("\n\n")
#%%


import matplotlib.pyplot as plt
import torch
y1h = torch.nn.functional.one_hot(y, num_classes=len(model.tokenizer.vocab)).float()

# no grid theme


plt.imshow(y1h[0].cpu().numpy().T, aspect="auto",interpolation="none")
plt.show()


# plot piano rolls,
# use a 16:9 aspect ratio for each plot
# subplots
fig, axs = plt.subplots(BATCH_SIZE,1, figsize=(4,2*BATCH_SIZE))
for i in range(BATCH_SIZE):        
    y_sm = model.tokenizer.decode(y[i])
    # print number of notes
    print(f"Number of notes: {y_sm.note_num()}")

    pr = piano_roll(y_sm, tpq=4)
    axs[i].imshow(pr, aspect="auto",interpolation="none")
plt.show()


preview_sm(y_sm)




 # %%



#%%
# get 5 highest
# for i in range(5):
#     y_sm = model.tokenizer.decode(y[idx[-i]])
#     print(f"Cross Entropy: {ce[idx[-i]]}")
#     print(f"Number of notes: {y_sm.note_num()}")
#     preview_sm(y_sm)
#     print("\n\n")
# %%
#%%

# get the embedding
embedding = model.embedding_layer.weight.detach().cpu().numpy().T
projection = model.decoder_output_layer.weight.detach().cpu().numpy()

# get vocab
vocab = model.tokenizer.vocab

embedding_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
projection_norm = np.linalg.norm(projection, axis=0, keepdims=True)

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