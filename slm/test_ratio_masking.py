#%%
from masking import ratio_superposition, random_superposition
import numpy as np
import torch
import matplotlib.pyplot as plt


ATTRIBUTE_VOCAB_SIZES = [3,9,27]
attribute_ranges = []
last_index = 0
for attribute_vocab_size in ATTRIBUTE_VOCAB_SIZES:
    attribute_ranges.append((last_index, last_index + attribute_vocab_size))
    last_index += attribute_vocab_size

N_EVENTS = 10

ATTRIBUTE_VOCABS = []

total_vocab_size = sum(ATTRIBUTE_VOCAB_SIZES)

print(total_vocab_size)

print(attribute_ranges)

# generate sequence of events for each attribute
x = np.stack(
    [ np.random.randint(low=attribute_ranges[a_idx][0], high=attribute_ranges[a_idx][1], size=N_EVENTS) for a_idx in range(len(ATTRIBUTE_VOCAB_SIZES)) ],
    axis=1
)

print(x.shape)
print(x)

# now one hot encode
events = np.eye(total_vocab_size)[x]

# syntax mask is a (N_ATTRIBUTES, VOCAB_SIZE) matrix where 1 means that the attribute is present in the vocab
# and 0 means it is not
syntax_mask = np.zeros((len(ATTRIBUTE_VOCAB_SIZES), total_vocab_size))
for a_idx, (start, end) in enumerate(attribute_ranges):
    syntax_mask[a_idx, start:end] = 1


# plot 
plt.figure()

plt.imshow(events.reshape(-1, total_vocab_size).T, interpolation="none")
plt.colorbar()
plt.show()

# plot syntax mask
plt.figure()
plt.imshow(syntax_mask.T, interpolation="none")
plt.colorbar()
plt.show()



# convert to tensors
x = torch.tensor(events, dtype=torch.float)[None,...]
syntax_mask = torch.tensor(syntax_mask, dtype=torch.float)

sup = random_superposition(x, syntax_mask)

plt.figure()
plt.imshow(sup[0].reshape(-1, total_vocab_size).T, interpolation="none")
plt.colorbar()
plt.show()

# %%
