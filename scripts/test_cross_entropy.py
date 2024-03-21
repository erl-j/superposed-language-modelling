#%%
import torch

VOCAB_SIZE = 100
target_idx = 10
target_logit = 1
# Create a random tensor (float)
x = torch.randn(VOCAB_SIZE)
x[target_idx] = target_logit

x2 = torch.randn(VOCAB_SIZE)
x2[target_idx] = target_logit

x3 = x.clone()
x3[target_idx] = target_logit * 2

loss_a = torch.nn.functional.cross_entropy(x.unsqueeze(0), torch.tensor([target_idx]))
loss_b = torch.nn.functional.cross_entropy(x2.unsqueeze(0), torch.tensor([target_idx]))
loss_c = torch.nn.functional.cross_entropy(x3.unsqueeze(0), torch.tensor([target_idx]))

# normalize the logits

print(f"Loss A: {loss_a}")
print(f"Loss B: {loss_b}")
print(f"Loss C: {loss_c}")

#%%

print(f"Loss: {loss}")
print(f"Loss with new target: {loss_new_target}")
print(f"Loss with same target: {loss_same_target}")

# %%
