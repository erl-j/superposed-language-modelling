#%%
from train import DecoderOnlyModel

#%%

device = "cuda:7"

# Load the model
# model = DecoderOnlyModel.load_from_checkpoint(
#     "../checkpoints/glamorous-water-14/epoch=14-step=220322-val/loss=0.35.ckpt"
# )

model = DecoderOnlyModel.load_from_checkpoint(
    "../checkpoints/cool-plasma-16/epoch=1-step=15529-val/loss=0.26-trn/loss=0.30.ckpt"
, map_location=device)

# Move the model to the device
model = model.to(device)

# Generate a sequence
a = model.tokenizer.get_format_mask()[None,...].to(model.device)


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
plt.figure(figsize=(40, 40))
# set small font
sns.set(font_scale=0.5)
sns.heatmap(similarity, cmap="magma", xticklabels=vocab, yticklabels=vocab)
plt.show()

#%%

# Generate a sequence
sequence = model.generate(a, max_len=model.tokenizer.total_len, temperature=0.99)
# %%

token_idx = sequence[0].cpu().numpy()

# argmax
token_idx = token_idx.argmax(axis=1)

# decode
sm = model.tokenizer.decode(token_idx)


# save the sequence
sm.dump_midi("../artefacts/generated.mid")

# %%
