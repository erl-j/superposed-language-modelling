#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from slm.train_old import EncoderOnlyModel
from util import preview, render_directory_with_fluidsynth
import os
import IPython.display as ipd
from slm.PAPER_CHECKPOINTS import checkpoints
from tqdm import tqdm
import torch

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

ROOT_DIR = "./"
TMP_DIR = ROOT_DIR + "tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/infilling_test"
device = "cuda:4"

def export_batch(y, tokenizer, output_dir):
    for sample_index in tqdm(range(y.shape[0])):
        # decode
        y_sm = tokenizer.decode(y[sample_index])
        out_path = output_dir + f"/nr_{sample_index}.mid"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        y_sm.dump_midi(out_path)
    render_directory_with_fluidsynth(output_dir, output_dir + "_audio/")

dummy_model = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + checkpoints["slm"],
        map_location=device,
    )
    .to(device)
    .eval()
)
# get tokenizer
tokenizer = dummy_model.tokenizer
del dummy_model

 
BATCH_SIZE = 1
MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR + "artefacts/val_midi_records_unique_pr.pt",
    path_filter_fn=lambda x: f"n_bars={MODEL_BARS}" in x,
    genre_list=tokenizer.config["tags"],
    tokenizer=tokenizer,
    min_notes=8 * MODEL_BARS,
    max_notes=tokenizer.config["max_notes"],
)

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
# get batch
batch = next(iter(dl))

# export batch
export_batch(batch,tokenizer,OUTPUT_DIR + "/natural")

#%%

infilling_tasks = [
"infilling_high",
"infilling_low",
"infilling_box",
"infilling_middle",
]

# infilling tasks
for infilling_task in infilling_tasks:
    for model_name in ["mlm","slm"]:
        for temperature in [0.85,0.95,0.99,1.0]:
            model = (
                EncoderOnlyModel.load_from_checkpoint(
                    ROOT_DIR + checkpoints[model_name],
                    map_location=device,
                )
                .to(device)
                .eval()
            )
            # prepare masks
            masks = []
            for sample_idx in range(batch.shape[0]):

                n_notes = tokenizer.decode(batch[sample_idx]).note_num()

                match infilling_task:
                    case "infilling_upper_half":
                        beat_range = (0,16)
                        pitch_range = [
                            f"pitch:{i}" for i in range(60,108)
                        ]
                    case "infilling_lower_half":
                        beat_range = (0,16)
                        pitch_range = [
                            f"pitch:{i}" for i in range(36,60)
                        ]
                    case "infilling_random_box":
                        beat_range = (4, 12)
                        pitch_range = [
                            f"pitch:{i}" for i in range(36,108)
                        ]
                    case "infilling_random_section":
                        beat_range = (4, 12)
                        pitch_range = None

                
                beat_range = (4, 12)
                pitch_range = [
                    f"pitch:{i}" for i in range(40,60)
                ]


                # make infilling mask
                mask = (
                    model.tokenizer.infilling_mask(
                        batch[sample_idx],
                        beat_range=beat_range,
                        min_notes=n_notes,
                        max_notes=n_notes,
                    )[None, ...]
                    .float()
                )

                masks.append(mask)

            mask = torch.cat(masks, dim=0).to(device)

            y = model.generate_batch(
                mask,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
            ).argmax(axis=-1)

            # get batch
            # export batch
            export_batch(y, model.tokenizer, OUTPUT_DIR + f"/{model_name}_t={temperature}")

#%%

resample_attribute_tasks = [
    "resample_pitches",
    "resample_timing",
]

# uses toms?

# dynamic drums

# unconstrained generation

# major pentatonic. guitar, drums, bass, rock beat

# loops stimuli per second.



constrained_generation_tasks = [
    "constrained_generation_a",
    "constrained_generation_b",
]


# Scenario : generation

# get N samples from tst set

# Scenario : constrained generation

# get N samples from tst set which abide by a constraint

# Scenario : infilling

# get N samples, use as input for infilling

# Scenario : resample pitches

# get N samples, use as input for resampling pitches

# Scenario : resample timing

# get N samples, use as input for resampling timing

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
