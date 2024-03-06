#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from merged_train import DecoderOnlyModel
from util import piano_roll

#%%

device = "cuda:7"

# Load the model
# model = DecoderOnlyModel.load_from_checkpoint(
#     "../checkpoints/glamorous-water-14/epoch=14-step=220322-val/loss=0.35.ckpt"
# )

model = DecoderOnlyModel.load_from_checkpoint(
    "../checkpoints/astral-dew-72/epoch=2-step=5341-val/loss=0.78-trn/loss=0.64.ckpt",
    map_location=device,
)

# Move the model to the device
model = model.to(device)

#%%
# Generate a sequence
a = model.format_mask[None,...].to(model.device)


# Generate a sequence
sequence = model.generate(a, max_len=model.tokenizer.total_len, temperature=0.95)

token_idx = sequence[0].cpu().numpy()

# argmax
token_idx = token_idx.argmax(axis=1)

#%%

# index to token
tokens = model.tokenizer.indices_to_tokens(token_idx)

print(tokens)

#%%


# decode
sm = model.tokenizer.decode(token_idx)

pr = piano_roll(sm)

plt.figure(figsize=(10, 10))
sns.heatmap(pr, cmap="magma")
plt.show()



# save the sequence
sm.dump_midi("../artefacts/generated.mid")

#%%

# plot embedding self similarity

import matplotlib.pyplot as plt
import seaborn as sns

# get the embedding
embedding = model.embedding_layer.weight.detach().cpu().numpy()

# get vocab
vocab = model.tokenizer.vocab

# compute the similarity matrix
similarity = embedding.T @ embedding

# plot the similarity matrix
# make large figure
plt.figure(figsize=(10, 10))
# set small font
sns.set(font_scale=0.5)
sns.heatmap(similarity, cmap="magma", xticklabels=vocab, yticklabels=vocab)
plt.show()

#%%


# %%
