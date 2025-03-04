#%%
from masking import ratio_superposition, random_superposition, mixed_superposition_2, simple_superposition
import numpy as np
import torch
import matplotlib.pyplot as plt


ATTRIBUTE_VOCAB_SIZES = [10,50,100]
attribute_ranges = []
last_index = 0
for attribute_vocab_size in ATTRIBUTE_VOCAB_SIZES:
    attribute_ranges.append((last_index, last_index + attribute_vocab_size))
    last_index += attribute_vocab_size

N_EVENTS = 50

ATTRIBUTE_VOCABS = []

total_vocab_size = sum(ATTRIBUTE_VOCAB_SIZES)

print(total_vocab_size)

print(attribute_ranges)

BATCH_SIZE=5000
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

#%%
# convert to tensors
x = torch.tensor(events, dtype=torch.float)
syntax_mask = torch.tensor(syntax_mask, dtype=torch.float)

schemes = {
    "simple": lambda x : simple_superposition(x, syntax_mask, superpositions = ["full","full", "full", "sparse", "shared", "shared_rate"], schedule_fn = lambda x: x**(1/4), attribute_masking_rate=0.05),
    "mixed" : lambda x : mixed_superposition_2(x)
}

plt.figure()

for name, scheme in schemes.items():

    sup = scheme(x)

    sup = sup * syntax_mask[None,...]

    # sup2 = sup + 0.3 * syntax_mask[None,...]
    # plt.figure()
    # plt.imshow(sup2[0].reshape(-1, total_vocab_size).T, interpolation="none")
    # plt.colorbar()
    # plt.show()


    # get some stats.

    # histogram of n_unkown (sum larger than 1)
    plt.hist((sup.sum(-1) > 1).sum(-1).sum(-1), bins=len(ATTRIBUTE_VOCAB_SIZES) * N_EVENTS, range=(0,len(ATTRIBUTE_VOCAB_SIZES) * N_EVENTS), alpha=0.2)
plt.savefig(f"figures/maskhist.png")



