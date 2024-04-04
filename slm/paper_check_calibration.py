
#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import preview
import os
import IPython.display as ipd
from paper_checkpoints import checkpoints
import torch

device = "cuda:4"
ROOT_DIR = "../"

MODEL = "slm"

OUTPUT_DIR = ROOT_DIR + "artefacts/examples_4"
TMP_DIR = ROOT_DIR + "artefacts/tmp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + checkpoints[MODEL],
        map_location=device,
    )
    .to(device)
    .eval()
)
m_records = []

BATCH_SIZE = 1

temperature = 1.0

for min_notes in range(1,300,50):
    for max_notes in range(1,300,50):
        if max_notes!=min_notes and max_notes >= min_notes:
            m_records.append({"min_notes": min_notes, "max_notes": max_notes})
            # a = model.format_mask[None, ...].to(model.device)
            # c = model.tokenizer.constraint_mask(
            #     # tags=["pop"],
            #     # tags=["other"],
            #     # instruments=["Drums", "Pipe", "Chromatic Percussion"],
            #     # tempos=["138"],
            #     # scale="G major",
            #     min_notes=min_notes,
            #     max_notes=max_notes,
            # )[None, ...].to(model.device)

            # c = c.repeat(BATCH_SIZE, 1, 1)

            # y = model.generate_batch(
            #     c,
            #     temperature=temperature,
            #     top_p=1.0,
            #     top_k=0,
            # ).argmax(axis=-1)

            # for idx in range(y.shape[0]):
            #     y_sm = model.tokenizer.decode(y[idx])
            #     records.append({""}

#%%
new_records = []
for i in range(len(m_records)):
    new_records.append({**m_records[i], "midi": records[i]})



#%%

new_records = [{**new_records[i], "n_notes":new_records[i]["midi"].note_num()} for i in range(len(m_records))]


#%%
#%%
# create heatmap with min notes on x, max notes on y and cell value as n_notes

min_note_values = list(set([record["min_notes"] for record in new_records]))
max_note_values = list(set([record["max_notes"] for record in new_records]))

# sort 
min_note_values.sort()
max_note_values.sort()
# create a matrix of zeros
mat = -np.ones((len(min_note_values), len(max_note_values)))
for record in new_records:
    min_note = record["min_notes"]
    max_note = record["max_notes"]
    n_notes = record["n_notes"]
    min_note_idx = min_note_values.index(min_note)
    max_note_idx = max_note_values.index(max_note)
    mat[min_note_idx, max_note_idx] = n_notes

# 

sns.heatmap(mat, annot=True, fmt=".0f", cmap="viridis", xticklabels=max_note_values, yticklabels=min_note_values)
plt.xlabel("max_notes")
plt.ylabel("min_notes")
plt.title("Number of notes in generated samples")
plt.show()





# %%
