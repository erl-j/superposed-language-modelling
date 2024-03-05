#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dense_train import UnetModel
from dense_transformer_train import DenseModel
from util import piano_roll

#%%

device = "cuda:2"

# Load the model
# model = DecoderOnlyModel.load_from_checkpoint(
#     "../checkpoints/glamorous-water-14/epoch=14-step=220322-val/loss=0.35.ckpt"
# )

# model = UnetModel.load_from_checkpoint(
#     "../checkpoints/unique-sun-17/epoch=9-step=69893-val/loss=0.08-trn/loss=0.08.ckpt",
#     map_location=device,
# )

# model = UnetModel.load_from_checkpoint(
#     "../checkpoints/fine-elevator-29/epoch=3-step=11895-val/loss=0.06-trn/loss=0.07.ckpt",
#     map_location=device,
# )

model = DenseModel.load_from_checkpoint(
    "../checkpoints/stellar-wood-46/epoch=0-step=6297-trn/loss=0.01.ckpt",
    map_location=device,
)
#%%

# Move the model to the device
model = model.to(device)


# Generate a sequence
a = model.tokenizer.get_format_mask()[None,...].to(model.device)


#%%

print(a.shape)
# plot format mask
plt.imshow(a[0,0, ...,0, :].cpu().numpy().T, aspect="auto", interpolation="none")
plt.show()

# plot format mask
plt.imshow(a[0,0, ...,1, :].cpu().numpy().T, aspect="auto", interpolation="none")
plt.show()

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

print(onsets.shape)

plt.imshow(onsets.cpu().detach().numpy().T, aspect="auto", interpolation="none")
plt.show()


#%%

print(a.shape)
s = model.generate(a, sampling_steps=100, temperature=0.95)

print(s.shape)

# argmax
s_idx = s.cpu().numpy().argmax(axis=-1)[0]


print(s_idx)

sm = model.tokenizer.decode(s_idx)
# save midi
sm.dump_midi("../artefacts/generated_dense.mid")

# print sm 
print(sm)

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

# Generate a sequence
sequence = model.generate(a, temperature=0.5)
# %%

token_idx = sequence[0].cpu().numpy()

# argmax
token_idx = token_idx.argmax(axis=1)

# decode
sm = model.tokenizer.decode(token_idx)


# save the sequence
sm.dump_midi("../artefacts/generated.mid")

# %%
