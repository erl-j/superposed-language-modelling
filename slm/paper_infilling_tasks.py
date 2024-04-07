#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import preview, render_directory_with_fluidsynth, get_sm_pitch_range, has_drum,has_harmonic
import os
import IPython.display as ipd
from paper_checkpoints import checkpoints
from tqdm import tqdm
import torch

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

ROOT_DIR = "./"
TMP_DIR = ROOT_DIR + "tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/eval/generate_tasks"
device = "cuda:6"

def export_batch(y, tokenizer, output_dir):
    for sample_index in tqdm(range(y.shape[0])):
        # decode
        y_sm = tokenizer.decode(y[sample_index])
        out_path = output_dir + f"/nr_{sample_index}.mid"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        y_sm.dump_midi(out_path)
    # render_directory_with_fluidsynth(output_dir, output_dir + "_audio")

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


BATCH_SIZE = 200
MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR + "artefacts/val_midi_records_unique_pr.pt",
    path_filter_fn=lambda x: f"n_bars={MODEL_BARS}" in x,
    genre_list=tokenizer.config["tags"],
    tokenizer=tokenizer,
    min_notes=8 * MODEL_BARS,
    max_notes=tokenizer.config["max_notes"],
    sm_filter_fn = lambda sm : has_drum(sm) and has_harmonic(sm)
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

tasks = [
# "infilling_high",
# "infilling_low",
# "infilling_start",
# "infilling_end"
# "infill_drums",
# "infilling_harmonic",
"generate",
]


# infilling tasks
for task in tasks:
    for model_name in ["mlm","slm"]:
        for temperature in [0.85,0.9,0.95,0.99,1.0]:
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

                sample_sm = tokenizer.decode(batch[sample_idx])
                n_notes = sample_sm.note_num()

                min_pitch,max_pitch = get_sm_pitch_range(sample_sm)

                if "infilling" in task: 

                    match task:
                        case "infilling_high":
                            min_box_pitch = (max_pitch - min_pitch)//2 + min_pitch
                            max_box_pitch = max_pitch 
                            pitches = [
                                f"pitch:{i}" for i in range(min_box_pitch,max_box_pitch)
                            ]
                            beat_range = (0,16)

                            
                        case "infilling_low":
                            min_box_pitch = min_pitch
                            max_box_pitch = (max_pitch - min_pitch)//2 + min_pitch
                            pitches = [
                                f"pitch:{i}" for i in range(min_box_pitch,max_box_pitch)
                            ]
                            beat_range = (0,16)

                        case "infilling_start":
                            beat_range = (0,8)
                        
                        case "infilling_end":
                            beat_range = (8,16)

                        case "infill_drums":
                            vocab = model.tokenizer.vocab
                            drum_range = [token for token in vocab if ("pitch" in token) and ("(Drums)" in token)]
                            pitches = drum_range
                            beat_range = (0,16)
                    
                        case "infill_harmonic":
                            pitches = [
                                f"pitch:{i}" for i in range(min_pitch,max_pitch)
                            ]
                            beat_range = (0,16)

                    # make infilling mask
                    mask = (
                        model.tokenizer.infilling_mask(
                            batch[sample_idx],
                            beat_range=beat_range,
                            pitches=pitches,
                            min_notes=n_notes,
                            max_notes=n_notes,
                        )[None, ...]
                        .float()
                    )
                
                elif "generate" in task:

                    mask = model.format_mask[None, ...].float()

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
            export_batch(y, model.tokenizer, OUTPUT_DIR + f"/{task}/{model_name}_t={temperature}")



