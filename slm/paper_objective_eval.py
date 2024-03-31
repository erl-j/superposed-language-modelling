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
from tqdm import tqdm


device = "cuda:7"
ROOT_DIR = "../"

MODEL = "slm"


model = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + checkpoints[MODEL],
        map_location=device,
    )
    .to(device)
    .eval()
)

if MODEL == "mlm":
    model.mlm_restricted_sampling = True

TMP_DIR = ROOT_DIR + "tmp"

OUTPUT_DIR = ROOT_DIR + "output"

#%%

N_EXAMPLES = 100

for i in tqdm(range(N_EXAMPLES)):
    a = model.format_mask[None, ...].to(model.device)

    # Generate a sequence
    y = model.generate(
        a,
        schedule_fn=lambda x: x,
        temperature=0.999,
        top_p=1.0,
        top_k=0,
    )[0].argmax(axis=1)

    # decode
    y_sm = model.tokenizer.decode(y)

    print(f"Number of notes: {y_sm.note_num()}")

    preview(y_sm, TMP_DIR)

    out_path = OUTPUT_DIR + "/generation/" + f"{MODEL}/example_{i}.mid"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    y_sm.dump_midi(out_path)

#%%



# tag prediction
# tempo prediction


# attribute prediction
    # pitch prediction
    # timing prediction

# reference: tst
# infilling
    # infilling high
    # infilling low
    # infilling box
    # infilling middle

# pitch set constraint
    # major, reference: things in major pitch set
    # pent, refernce: things in pentatonic pitch set

# raw generation

# instrument constraint
    # drums, bass, piano, guitar
