#%%

device = "cuda:3"
from bfn import BFNModel

checkpoint = "../checkpoints/wobbly-river-40/epoch=0-step=10375-val/loss_epoch=0.01564.ckpt"

model = BFNModel.load_from_checkpoint(checkpoint, map_location=device)
# %%



tokenizer = model.tokenizer

y = model.sample(10,20,device=device, plot_interval=5)

import matplotlib.pyplot as plt
import torch
y1h = torch.nn.functional.one_hot(y, num_classes=len(model.tokenizer.vocab)).float()

plt.imshow(y1h[0].cpu().numpy().T, aspect="auto",interpolation="none")
plt.show()

from util import preview

y_sm = model.tokenizer.decode(y[0])

print(f"number of notes: {y_sm.note_num()}")

preview(y_sm, "artefacts/tmp.mid")
# %%
