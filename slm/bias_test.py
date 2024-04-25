#%%
import torch

V=20

prior = torch.ones(V)

prior[:V//2] = 0

# normalize
prior = prior/prior.sum()




# %%

# uniform prior
uniform = torch.ones(V)/V

print(prior.shape)


for ls in [0.0,0.2,0.5,0.8,1.0]:
    mix = (1-ls)*uniform + (ls)*prior
    # y lim is 0 to 1
    plt.scatter(range(V), mix, label=f"ls={ls}")
    plt.ylim(0,1)
    plt.title(f"ls={ls}")
    plt.show()
# %%
