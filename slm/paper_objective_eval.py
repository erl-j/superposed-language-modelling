#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import preview, render_directory_with_fluidsynth
import os
import IPython.display as ipd
from paper_checkpoints import checkpoints
from tqdm import tqdm
import torch


def export_batch(y, model, output_dir):
    for sample_index in tqdm(range(y.shape[0])):
        # decode
        y_sm = model.tokenizer.decode(y[sample_index])

        out_path = output_dir + f"/nr_{sample_index}.mid"

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        y_sm.dump_midi(out_path)

    render_directory_with_fluidsynth(output_dir, output_dir + "_audio/")


SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


device = "cuda:4"
ROOT_DIR = "../"

MODEL = "mlm"


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

OUTPUT_DIR = ROOT_DIR + "artefacts/object_eval"

#%%
BATCH_SIZE = 64

MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR + "artefacts/tst_midi_records_unique_pr.pt",
    path_filter_fn=lambda x: f"n_bars={MODEL_BARS}" in x,
    genre_list=model.tokenizer.config["tags"],
    tokenizer=model.tokenizer,
    min_notes=8 * MODEL_BARS,
    max_notes=model.tokenizer.config["max_notes"],
)

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# get batch
batch = next(iter(dl))

# export batch
export_batch(batch, model, OUTPUT_DIR + "/natural")

#%%

TEMPERATURE = 1.0

a = model.format_mask[None, ...].to(model.device)

# repeat a to match batch size
a = a.repeat(BATCH_SIZE, 1, 1)

# Generate a sequence
y = model.generate_batch(
    a,
    temperature=TEMPERATURE,
    top_p=1.0,
    top_k=0,
).argmax(axis=-1)

# export batch
export_batch(y, model, OUTPUT_DIR + f"/{MODEL}_temp_{TEMPERATURE}")


#%%



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
