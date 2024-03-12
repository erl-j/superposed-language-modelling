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

# model = DecoderOnlyModel.load_from_checkpoint(
#     "../checkpoints/astral-dew-72/epoch=33-step=62817-val/loss=0.60-trn/loss=0.60.ckpt",
#     map_location=device,
# )

# model = DecoderOnlyModel.load_from_checkpoint(
#     "../checkpoints/scarlet-serenity-114/epoch=14-step=63352-val/loss=0.29-trn/loss=0.32.ckpt",
#     map_location=device,
# )

# model = DecoderOnlyModel.load_from_checkpoint(
#     "../checkpoints/azure-frog-129/epoch=20-step=52701-val/loss=0.32-trn/loss=0.32.ckpt",
#     map_location=device,
# )

model = DecoderOnlyModel.load_from_checkpoint(
    "../checkpoints/azure-frog-129/epoch=29-step=77024-val/loss=0.31-trn/loss=0.26.ckpt",
    map_location=device,
)

# Move the model to the device
model = model.to(device)

#%%
# Generate a sequence
a = model.format_mask[None,...].to(model.device)

# print(model.tokenizer.tempo_bins)

c = model.tokenizer.constraint_mask(
    # tags=["classical"],
    instruments=["Piano","Drums"],
    tempos = ["126"],
    scale = "C major",
    min_notes=30,
)[None,...].to(model.device)
a = c*a
plt.figure(figsize=(10, 60))
sns.heatmap(a[0][:15].cpu().numpy().T, cmap="magma", yticklabels=model.tokenizer.vocab)
plt.show()

#%%


print(a.shape)

# Generate a sequence
sequence = model.generate(a,
                        max_len=model.tokenizer.total_len, 
                        temperature=1.0,
                        top_p=1.0,
                          top_k=0,
                        )


token_idx = sequence[0].cpu().numpy()

# argmax
token_idx = token_idx.argmax(axis=1)


# index to token
tokens = model.tokenizer.indices_to_tokens(token_idx)

# chunk into attributes
tokens = [tokens[i:i+model.tokenizer.attributes_per_note] for i in range(0, len(tokens), model.tokenizer.attributes_per_note)]

print(tokens)

# decode
sm = model.tokenizer.decode(token_idx)

# print number of notes
print(sm.note_num())


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
