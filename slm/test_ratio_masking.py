#%%
from masking import ratio_superposition, random_superposition, mixed_superposition_2, simple_superposition
import numpy as np
import torch
import matplotlib.pyplot as plt


ATTRIBUTE_VOCAB_SIZES = [3,9,27]
attribute_ranges = []
last_index = 0
for attribute_vocab_size in ATTRIBUTE_VOCAB_SIZES:
    attribute_ranges.append((last_index, last_index + attribute_vocab_size))
    last_index += attribute_vocab_size

N_EVENTS = 30

ATTRIBUTE_VOCABS = []

total_vocab_size = sum(ATTRIBUTE_VOCAB_SIZES)

print(total_vocab_size)

print(attribute_ranges)

BATCH_SIZE=1000
# generate sequence of events for each attribute
x = np.stack(
    [ np.random.randint(low=attribute_ranges[a_idx][0], high=attribute_ranges[a_idx][1], size=(BATCH_SIZE,N_EVENTS)) for a_idx in range(len(ATTRIBUTE_VOCAB_SIZES)) ],
    axis=2
)

print(x.shape)
print(x)

#%%

# now one hot encode
events = np.eye(total_vocab_size)[x]

# syntax mask is a (N_ATTRIBUTES, VOCAB_SIZE) matrix where 1 means that the attribute is present in the vocab
# and 0 means it is not
syntax_mask = np.zeros((len(ATTRIBUTE_VOCAB_SIZES), total_vocab_size))
for a_idx, (start, end) in enumerate(attribute_ranges):
    syntax_mask[a_idx, start:end] = 1

#%%
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
x = torch.tensor(events, dtype=torch.float)
syntax_mask = torch.tensor(syntax_mask, dtype=torch.float)

# sup = ratio_superposition(x, syntax_mask, superpositions=["full", "full", "sparse","shared_rate"], schedule_fn= lambda x: x**(1/4), simulate_autoregression=False)
# sup = ratio_superposition(x, syntax_mask, superpositions=["full", "full", "sparse","shared_rate"], simulate_autoregression=True)
# sup = mixed_superposition_2(x)
sup = simple_superposition(x, syntax_mask, superpositions = ["full","sparse"], schedule_fn = lambda x: x**(1/4), attribute_masking_rate=0.05)
# multiply by syntax mask
sup = sup * syntax_mask[None,...]
print(sup.shape)

sup2 = sup + 0.3 * syntax_mask[None,...]
plt.figure()
plt.imshow(sup2[0].reshape(-1, total_vocab_size).T, interpolation="none")
plt.colorbar()
plt.show()

#%%

# get some stats.

# histogram of n_unkown (sum larger than 1)
plt.figure()
plt.hist((sup.sum(-1) > 1).sum(-1).sum(-1), bins=len(ATTRIBUTE_VOCAB_SIZES) * N_EVENTS)
plt.show()

# plot n_unkown histogram for each attribute
print(sup.shape)
plt.figure()
for attr_idx in range(sup.shape[2]):
    plt.hist((sup[:,:,attr_idx].sum(-1) > 1).sum(-1))
    plt.show()


# %%
