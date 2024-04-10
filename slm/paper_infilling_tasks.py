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
import einops

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

ROOT_DIR = "./"
TMP_DIR = ROOT_DIR + "tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/eval/new_tasks"
device = "cuda:3"

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


BATCH_SIZE = 100
MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR + "paper_assets/tst_midi_records_unique_pr.pt",
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
batch = next(iterator)

if False:

    # export batch
    export_batch(batch,tokenizer,OUTPUT_DIR + "/natural")

    # get second batch, different from the first
    batch2 = next(iterator)

    # assert that the two batches are different
    assert not torch.allclose(batch, batch2)

    export_batch(batch2,tokenizer,OUTPUT_DIR + "/natural2")



#%%

tasks = [
# "generate",
# "infilling_low",
"pitch_set",
"infilling_high_patched",
# "infilling_start",
# "infilling_end",
"infilling_box_middle",
"infilling_box_end",
# "constrained_generation"
"onset_set",
"pitch_onset_set",
"infilling_drums",
"infilling_harmonic",
# ,"constrained_generation",
]

# infilling tasks
for temperature in [0.95,1.0,1.05]:
    for task in tasks:
        for model_name in ["mlm","slm"]:
            model = (
                EncoderOnlyModel.load_from_checkpoint(
                    ROOT_DIR + checkpoints[model_name],
                    map_location=device,
                )
                .to(device)
                .eval()
            )
            print(device)
            # prepare masks
            masks = []
            for sample_idx in range(batch.shape[0]):

                sample_sm = tokenizer.decode(batch[sample_idx])
                n_notes = sample_sm.note_num()

                min_pitch,max_pitch = get_sm_pitch_range(sample_sm)

                if "infilling" in task: 

                    match task:
                        case "infilling_high_patched":
                            min_box_pitch = (max_pitch - min_pitch)//2 + min_pitch
                            max_box_pitch = max_pitch 
                            pitches = [
                                f"pitch:{i}" for i in range(min_box_pitch,max_box_pitch+1)
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
                            pitches=None
                        
                        case "infilling_end":
                            beat_range = (8,16)
                            pitches=None

                        case "infilling_drums":
                            vocab = model.tokenizer.vocab
                            drum_range = [token for token in vocab if ("pitch:" in token) and ("(Drums)" in token)]
                            pitches = drum_range
                            beat_range = (0,16)
                    
                        case "infilling_harmonic":
                            pitches = [
                                f"pitch:{i}" for i in range(min_pitch,max_pitch+1)
                            ]
                            beat_range = (0,16)

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

                        case "infilling_box_end":
                            # pitch range 
                            pitches = [

                                f"pitch:{i}" for i in range(min_pitch,max_pitch+1)
                            ]
                            min_box_pitch = (max_pitch - min_pitch)//2 + min_pitch
                            max_box_pitch = max_pitch 
                            pitches = [
                                f"pitch:{i}" for i in range(min_box_pitch,max_box_pitch+1)
                            ]
                            beat_range = (12,16)

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

                elif task == "onset_beats":
                    x = batch[sample_idx]
                    x_sm = model.tokenizer.decode(x)
                    n_notes = x_sm.note_num()

                    x1h = torch.nn.functional.one_hot(x, num_classes=len(model.tokenizer.vocab)).float()

                    # reshape x have (n_notes,n_attributes,vocab)
                    x_a = einops.rearrange(x1h,"(n_notes n_attributes) vocab -> n_notes n_attributes vocab",vocab=len(model.tokenizer.vocab),n_attributes=len(model.tokenizer.note_attribute_order))
                    # get pitch tokens
                    onset_tokens = (x_a[:,model.tokenizer.note_attribute_order.index("onset/beat"),:].sum(axis=0)>0 ).float()
                    x_a[:,model.tokenizer.note_attribute_order.index("onset/beat"),:] = onset_tokens
                    mask = einops.rearrange(x_a,"n_notes n_attributes vocab -> (n_notes n_attributes) vocab")[None,...]

                elif task == "pitch_onset_set":

                    x = batch[sample_idx]
                    x_sm = model.tokenizer.decode(x)
                    n_notes = x_sm.note_num()

                    x1h = torch.nn.functional.one_hot(x, num_classes=len(model.tokenizer.vocab)).float()

                    # reshape x have (n_notes,n_attributes,vocab)
                    x_a = einops.rearrange(x1h,"(n_notes n_attributes) vocab -> n_notes n_attributes vocab",vocab=len(model.tokenizer.vocab),n_attributes=len(model.tokenizer.note_attribute_order))
                    # get pitch tokens
                    pitch_tokens = (x_a[:,model.tokenizer.note_attribute_order.index("pitch"),:].sum(axis=0)>0).float()
                    x_a[:,model.tokenizer.note_attribute_order.index("pitch"),:] = pitch_tokens

                    onset_tokens = (x_a[:,model.tokenizer.note_attribute_order.index("onset/beat"),:].sum(axis=0)>0).float()
                    x_a[:,model.tokenizer.note_attribute_order.index("onset/beat"),:] = onset_tokens

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



