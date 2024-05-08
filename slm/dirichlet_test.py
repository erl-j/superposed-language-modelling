#%%
from dirichlet_flow import DirichletFlowModel

device = 'cuda:0'

ckpt = "../checkpoints/zany-thunder-48/last.ckpt"
model = DirichletFlowModel.load_from_checkpoint(
    ckpt, device=device
).to(device)



inference_args = model.flow_args
print(inference_args)
inference_args.num_integration_steps = 2000
inference_args.flow_temp = 0.7
inference_args.alpha_spacing = 0.02
# inference_args.alpha_max = 10.0
# inference_args.alpha_scale = 2.0
# inference_args.alpha_max = 30.0
# inference_args.alpha_scale = 2.0



l,x = model.generate(
    1,
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
