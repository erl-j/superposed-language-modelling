#%%
device = "cuda:0"
from simplex_diffusion import SimplexDiffusionModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from util import preview_sm,piano_roll
import matplotlib.pyplot as plt
import seaborn as sns
from data import MidiDataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import preview, render_directory_with_fluidsynth, has_drum, has_harmonic, get_sm_pitch_range
import os
import IPython.display as ipd
from paper_checkpoints import checkpoints
from tqdm import tqdm
import torch
import einops

#%%
checkpoint = "../checkpoints/fanciful-planet-7/last.ckpt"
# checkpoint = "../checkpoints/super-mountain-5/last.ckpt"
checkpoint = "../checkpoints/effortless-resonance-33/last.ckpt"
checkpoint = "../checkpoints/serene-sunset-44/last.ckpt"
# checkpoint = "../checkpoints/driven-violet-62/last.ckpt"
model = SimplexDiffusionModel.load_from_checkpoint(checkpoint, map_location=device)

#%%
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

ROOT_DIR = "./"
TMP_DIR = ROOT_DIR + "tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/simplex/"
device = "cuda:4"


SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

ROOT_DIR = "./"
TMP_DIR = ROOT_DIR + "tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/simplex/"
device = "cuda:0"

def export_batch(y, tokenizer, output_dir):
    for sample_index in tqdm(range(y.shape[0])):
        # decode
        y_sm = tokenizer.decode(y[sample_index])
        out_path = output_dir + f"/nr_{sample_index}.mid"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        y_sm.dump_midi(out_path)

# get tokenizer
tokenizer = model.tokenizer

BATCH_SIZE = 10
MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR + "data/mmd_loops/tst_midi_records_unique_pr.pt",
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
iterator = iter(dl)
# get batch
# drop first
batch = next(iterator)

export_batch(batch,tokenizer,OUTPUT_DIR + "/natural")

#%%

tasks = [
"pitch_set",
"onset_offset_set",
"pitch_onset_offset_set",
"infilling_box_middle",
"infilling_high", 
"infilling_low",
"infilling_start",
"infilling_end",
"generate",
"resample_velocity",
"variation"
"constrained_generation"
"infilling_harmonic",
"infilling_drums"
]

# infilling tasks
for task in tasks:
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
                            f"pitch:{i}" for i in range(min_box_pitch,max_box_pitch+1)
                        ]
                        beat_range = (0,16)
                        infilling_mode = "harmonic"

                        
                    case "infilling_low":
                        min_box_pitch = min_pitch
                        max_box_pitch = (max_pitch - min_pitch)//2 + min_pitch
                        pitches = [
                            f"pitch:{i}" for i in range(min_box_pitch,max_box_pitch)
                        ]
                        beat_range = (0,16)
                        infilling_mode = "harmonic"

                    case "infilling_start":
                        beat_range = (0,8)
                        pitches=None
                        infilling_mode = "harmonic+drums"
                    
                    case "infilling_end":
                        beat_range = (8,16)
                        pitches=None
                        infilling_mode = "harmonic+drums"

                    case "infilling_drums":
                        vocab = model.tokenizer.vocab
                        drum_range = [token for token in vocab if ("pitch:" in token) and ("(Drums)" in token)]
                        pitches = drum_range
                        beat_range = (0,16)
                        infilling_mode = "drums"
                
                    case "infilling_harmonic":
                        pitches = [
                            f"pitch:{i}" for i in range(min_pitch,max_pitch+1)
                        ]
                        beat_range = (0,16)
                        infilling_mode = "harmonic"

                    case "infilling_box_middle":
                        # pitch range 
                        pitches = [

                            f"pitch:{i}" for i in range(min_pitch,max_pitch+1)
                        ]
                        min_box_pitch = (max_pitch - min_pitch)//2 + min_pitch
                        max_box_pitch = max_pitch 
                        pitches = [
                            f"pitch:{i}" for i in range(min_box_pitch,max_box_pitch+1)
                        ]
                        beat_range = (8,12)
                        infilling_mode = "harmonic"

                # make infilling mask
                mask = (
                    model.tokenizer.infilling_mask(
                        batch[sample_idx],
                        beat_range=beat_range,
                        pitches=pitches,
                        min_notes=n_notes,
                        max_notes=n_notes,
                        mode=infilling_mode
                    )[None, ...]
                    .float()
                ) 
            
            elif "generate" in task:

                mask = model.format_mask[None, ...].float()

            elif "constrained" in task:
                x = batch[sample_idx]
                x_sm = model.tokenizer.decode(x)
                x_tokens = model.tokenizer.indices_to_tokens(x)

                pitch_tokens = [token for token in x_tokens if "pitch:" in token and ":-" not in token]
                # take unique
                pitch_tokens = list(set(pitch_tokens))
                instrument_tokens = [token for token in x_tokens if ("instrument:" in token and ":-" not in token)]
                instruments = [token.split(":")[1] for token in instrument_tokens]
                # take unique
                instruments = list(set(instruments))

                tempo_tokens = [token for token in x_tokens if "tempo:" in token]
                tempos = [token.split(":")[1] for token in tempo_tokens if ":-" not in token]
                tempos = list(set(tempos))
                tag_tokens = [token for token in x_tokens if "tag:" in token if ":-" not in token]
                tags = [token.split(":")[1] for token in tag_tokens]
                tags = list(set(tags))

                # onset beats
                onset_beat_tokens = [token for token in x_tokens if "onset/beat:" in token if ":-" not in token]
                # get unique
                onset_beat_tokens = list(set(onset_beat_tokens))

                n_notes = x_sm.note_num()

                mask = model.tokenizer.constraint_mask(
                    pitches = pitch_tokens,
                    instruments = instruments,
                    onset_beats=onset_beat_tokens,
                    tempos = tempos,
                    # tags = tags,
                    min_notes=n_notes,
                    max_notes=n_notes,
                    min_notes_per_instrument=1,
                )[None, ...].float()

            elif task == "pitch_set":
                x = batch[sample_idx]
                x_sm = model.tokenizer.decode(x)
                n_notes = x_sm.note_num()

                x1h = torch.nn.functional.one_hot(x, num_classes=len(model.tokenizer.vocab)).float()

                # reshape x have (n_notes,n_attributes,vocab)
                x_a = einops.rearrange(x1h,"(n_notes n_attributes) vocab -> n_notes n_attributes vocab",vocab=len(model.tokenizer.vocab),n_attributes=len(model.tokenizer.note_attribute_order))
                # get pitch tokens
                pitch_tokens = (x_a[:,model.tokenizer.note_attribute_order.index("pitch"),:].sum(axis=0)>0).float()
                # replace pitch attribute with pitch tokens, making sure that shape matches x_a
                print(pitch_tokens.shape)
                x_a[:,model.tokenizer.note_attribute_order.index("pitch")] = pitch_tokens
                mask = einops.rearrange(x_a,"n_notes n_attributes vocab -> (n_notes n_attributes) vocab")[None,...]

            elif task == "onset_offset_set":
                x = batch[sample_idx]
                x_sm = model.tokenizer.decode(x)
                n_notes = x_sm.note_num()

                x1h = torch.nn.functional.one_hot(x, num_classes=len(model.tokenizer.vocab)).float()

                # reshape x have (n_notes,n_attributes,vocab)
                x_a = einops.rearrange(x1h,"(n_notes n_attributes) vocab -> n_notes n_attributes vocab",vocab=len(model.tokenizer.vocab),n_attributes=len(model.tokenizer.note_attribute_order))
                # get pitch tokens
                onset_tokens = (x_a[:,model.tokenizer.note_attribute_order.index("onset/beat"),:].sum(axis=0)>0).float()
                x_a[:,model.tokenizer.note_attribute_order.index("onset/beat")] = onset_tokens

                # resample offsets
                offset_tokens = (x_a[:,model.tokenizer.note_attribute_order.index("offset/beat"),:].sum(axis=0)>0).float()
                x_a[:,model.tokenizer.note_attribute_order.index("offset/beat")] = offset_tokens
                # get pitch tokens

                mask = einops.rearrange(x_a,"n_notes n_attributes vocab -> (n_notes n_attributes) vocab")[None,...]

            elif task == "pitch_onset_offset_set":

                x = batch[sample_idx]
                x_sm = model.tokenizer.decode(x)
                n_notes = x_sm.note_num()

                x1h = torch.nn.functional.one_hot(x, num_classes=len(model.tokenizer.vocab)).float()

                # reshape x have (n_notes,n_attributes,vocab)
                x_a = einops.rearrange(x1h,"(n_notes n_attributes) vocab -> n_notes n_attributes vocab",vocab=len(model.tokenizer.vocab),n_attributes=len(model.tokenizer.note_attribute_order))
                # get pitch tokens

                # get pitch tokens
                pitch_tokens = (x_a[:,model.tokenizer.note_attribute_order.index("pitch"),:].sum(axis=0)>0).float()
                x_a[:,model.tokenizer.note_attribute_order.index("pitch")] = pitch_tokens

                # get pitch tokens
                onset_tokens = (x_a[:,model.tokenizer.note_attribute_order.index("onset/beat"),:].sum(axis=0)>0).float()
                x_a[:,model.tokenizer.note_attribute_order.index("onset/beat")] = onset_tokens

                # resample offsets
                offset_tokens = (x_a[:,model.tokenizer.note_attribute_order.index("offset/beat"),:].sum(axis=0)>0).float()
                x_a[:,model.tokenizer.note_attribute_order.index("offset/beat")] = offset_tokens


                mask = einops.rearrange(x_a,"n_notes n_attributes vocab -> (n_notes n_attributes) vocab")[None,...]

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



