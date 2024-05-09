#%%
from dirichlet_flow import DirichletFlowModel
import torch
device = 'cuda:0'

# ckpt = "../checkpoints/zany-thunder-48/last.ckpt"
ckpt = "../checkpoints/leafy-wood-51/last.ckpt"
model = DirichletFlowModel.load_from_checkpoint(
    ckpt, device=device
).to(device)



inference_args = model.flow_args
print(inference_args)
inference_args.num_integration_steps = 200
inference_args.flow_temp = 0.8
inference_args.alpha_spacing = 0.2
inference_args.alpha_max = 8.0
# inference_args.alpha_max = 30.0
# inference_args.alpha_scale = 2.0
# inference_args.alpha_max = 30.0
# inference_args.alpha_scale = 2.0


mask = model.tokenizer.constraint_mask(
    instruments=["Piano"],
    min_notes=30,
    max_notes=200,
    min_notes_per_instrument=5
)

mask = torch.tensor(mask * model.tokenizer.get_format_mask()).float()

print(mask.shape)

prior = mask / mask.sum(dim=-1, keepdim=True)[None,:]

l,x = model.generate(
    prior,
    inference_args,
)


xs = x.argmax(dim=-1)

import matplotlib.pyplot as plt
plt.imshow(x[0].cpu().numpy(), cmap='gray', aspect='auto', origin='lower')
plt.show()


plt.plot(x[0,0].cpu().numpy())
plt.show()


x_sm = model.tokenizer.decode(xs[0].cpu().numpy())


from util import preview_sm

preview_sm(x_sm)



# %%
