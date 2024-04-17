#%%

device = "cuda:3"
from bfn import BFNModel

checkpoint = "../checkpoints/wobbly-river-40/epoch=2-step=31125-val/loss_epoch=0.00729.ckpt"

model = BFNModel.load_from_checkpoint(checkpoint, map_location=device)
# %%

tokenizer = model.tokenizer

mask = tokenizer.constraint_mask(
    scale="C major",
    instruments = ["Piano","Drums","Bass"],
    min_notes = 50,
    max_notes = 100,
)


BATCH_SIZE = 10
N_STEPS = 20
y = model.sample(None,BATCH_SIZE,N_STEPS,device=device,argmax=False)

import matplotlib.pyplot as plt
import torch
y1h = torch.nn.functional.one_hot(y, num_classes=len(model.tokenizer.vocab)).float()

plt.imshow(y1h[0].cpu().numpy().T, aspect="auto",interpolation="none")
plt.show()

from util import preview, piano_roll

# plot piano rolls,
# use a 16:9 aspect ratio for each plot
# subplots
fig, axs = plt.subplots(BATCH_SIZE,1, figsize=(4,2*BATCH_SIZE))
for i in range(BATCH_SIZE):        
    y_sm = model.tokenizer.decode(y[i])
    # print number of notes
    print(f"Number of notes: {y_sm.note_num()}")

    pr = piano_roll(y_sm, tpq=4)
    axs[i].imshow(pr, aspect="auto",interpolation="none")
plt.show()

# play audio of last 
preview(y_sm, tmp_dir="tmp", audio=True)
    
