#%%
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
from util import preview, render_directory_with_fluidsynth, has_drum, has_harmonic, get_sm_pitch_range, load_merged_models, sm_fix_overlap_notes
import os
import IPython.display as ipd
from paper_checkpoints import checkpoints
from tqdm import tqdm
import torch
import einops

#%%
ROOT_DIR = "../"
device = "cuda:0"

# print model
# model = load_merged_models(ROOT_DIR+"checkpoints/dark-sky-67/**/*.ckpt",SimplexDiffusionModel).to(device)
#model = load_merged_models(ROOT_DIR+"checkpoints/flowing-paper-64/**/*.ckpt",SimplexDiffusionModel).to(device)
# get hidden size
model = SimplexDiffusionModel.load_from_checkpoint(ROOT_DIR+"checkpoints/dark-sky-67/last.ckpt", map_location=device)
#model = SimplexDiffusionModel.load_from_checkpoint(ROOT_DIR+"checkpoints/flowing-paper-64/last.ckpt", map_location=device)
hidden_size = model.hparams.hidden_size
model.eval()
#%%
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


TMP_DIR = ROOT_DIR + "tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/simplex_demo_3_/"
DEMO_DIR = ROOT_DIR + "artefacts/simplex_demo_3_website/"

def export_batch(y, tokenizer, output_dir):
    for sample_index in tqdm(range(y.shape[0])):
        # decode
        y_sm = tokenizer.decode(y[sample_index])
        out_path = output_dir + f"/nr_{sample_index}.mid"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        y_sm.dump_midi(out_path)

# get tokenizer
tokenizer = model.tokenizer

MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR + "data/mmd_loops/tst_midi_records_unique_pr.pt",
    path_filter_fn=lambda x: f"n_bars={MODEL_BARS}" in x,
    genre_list=tokenizer.config["tags"],
    tokenizer=tokenizer,
    min_notes=8 * MODEL_BARS,
    max_notes=tokenizer.config["max_notes"],
    # sm_filter_fn = lambda sm : has_drum(sm) and has_harmonic(sm)
)


#%%

#likes = [317, 762, 2705, 4868, 6033, 7165, 8118, 10730, 12045, 12085, 12843, 12989, 13633, 13819, 13823, 15112, 15790, 16326, 17115, 18310, 18424, 18619, 21862]
likes = [430, 1403,50,280,327, 420 ]

BATCH_SIZE = len(likes)
curated_test_data = likes
# curated_test_data.sort()

# create batch from likes
batch = torch.stack([ds[index] for index in curated_test_data], dim=0)

export_batch(batch,tokenizer,OUTPUT_DIR + "/natural")
export_batch(batch,tokenizer,DEMO_DIR + "/natural")

# preview batch
for sample_index in range(batch.shape[0]):
    print(f"likes: {curated_test_data[sample_index]}")
    # decode
    sm = tokenizer.decode(batch[sample_index])
    preview_sm(sm)

#%%

tasks = [
# "generate",
"generate_w_constraints",
#"pitch_set", # 100 0.5 True 1.0 True
# "onset_offset_set",
#"pitch_onset_offset_set",
# "infilling_box_middle",
#"infilling_high", # 50 0.0 TRUE 1.0 / 100, 0.85, True, 1.0 , True / 200 ,0.8, True, 1.0, True
#"infilling_middle", # 50 , 0.0, TRUE, 1.0 / 200, 0.75, TRUE, 1.0 / 200 ,0.8, True, 1.0, True
#"infilling_low", # 50, 0.0, False, 1.0 / 200, 0.75, True, 1.0 
# "infilling_start",
# "infilling_end",
# "variation",
#"infilling_harmonic", #  200 0.85 True 1.0 True
#"infilling_drums" # 200, 0.85, True, 1.0, True
# "replace_bass",
#"replace_chromatic",
# "replace_drums" d
]

# 200 0.75/0.85 True 1.0

# GRID_STEPS = [5,10,25,50,100,200]
# GRID_TOPP = [0.0,0.5,0.8,0.9,0.95,0.99,1.0]

GRID_STEPS = [50]
GRID_TOPP = [1.0]

PRIOR_STRENGTH = 1.0
ENFORCE_PRIOR = True
MULTIPLY_PRIOR = True


# infilling tasks
for task in tasks:
    for N_STEPS in GRID_STEPS:
        for TOPP in GRID_TOPP:

            prior_strength = PRIOR_STRENGTH
            
            print(f"Task: {task}")
            print(f"##################")
            
            enforce_prior = True

            # prepare masks
            masks = []
            for sample_idx in range(batch.shape[0]):

                sample_sm = tokenizer.decode(batch[sample_idx])
                n_notes = sample_sm.note_num()

                min_pitch,max_pitch = get_sm_pitch_range(sample_sm)

                if "infilling" in task: 

                    mode = ".."

                    top_p = TOPP

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

                        case "infilling_middle":
                            beat_range = (4,12)
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
                
                elif "generate" == task:

                    top_p = TOPP
                    enforce_prior = True

                    # mask = model.tokenizer.constraint_mask(
                    #     # tags=["jazz"],
                    #     scale="C major",
                    #     instruments=["Piano","Bass","Drums"],
                    #     # tempos=["138"],
                    #     min_notes=25,
                    #     max_notes=275,
                    #     min_notes_per_instrument=20
                    # )[None, ...].float()

                    mask = model.format_mask[None, ...].float()

                elif "generate_w_constraints" in task:

                    top_p = TOPP
                    enforce_prior = True

                    mask = model.tokenizer.constraint_mask(
                        tags=["pop"],
                        scale="C major",
                        instruments=["Piano","Bass","Drums","Guitar"],
                        tempos=["180"],
                        min_notes=50,
                        max_notes=250,
                        min_notes_per_instrument=10
                    )[None, ...].float()

                elif "constrained" in task:

                    top_p = TOPP
                    enforce_prior = True

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
                        min_notes_per_instrument=10,
                    )[None, ...].float()

                elif task == "pitch_set":
                    top_p = TOPP
                    enforce_prior = True

                    x = batch[sample_idx]
                    x_sm = model.tokenizer.decode(x)
                    n_notes = x_sm.note_num()

                    x1h = torch.nn.functional.one_hot(x, num_classes=len(model.tokenizer.vocab)).float()

                    # reshape x have (n_notes,n_attributes,vocab)
                    x_a = einops.rearrange(x1h,"(n_notes n_attributes) vocab -> n_notes n_attributes vocab",vocab=len(model.tokenizer.vocab),n_attributes=len(model.tokenizer.note_attribute_order))
                    # get pitch tokens
                    pitch_tokens = (x_a[:,model.tokenizer.note_attribute_order.index("pitch"),:].sum(axis=0)>0).float()
                    # replace pitch attribute with pitch tokens, making sure that shape matches x_a
                    x_a[:,model.tokenizer.note_attribute_order.index("pitch")] = pitch_tokens
                    mask = einops.rearrange(x_a,"n_notes n_attributes vocab -> (n_notes n_attributes) vocab")[None,...]

                elif task == "onset_offset_set":
                    top_p = TOPP
                    enforce_prior = True

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

                    # # blank out onset / offset ticks
                    # x_a[:,model.tokenizer.note_attribute_order.index("offset/tick")] = 1
                    # x_a[:,model.tokenizer.note_attribute_order.index("onset/tick")] = 1
                    # # blank out onset / offset beats
                    # x_a[:,model.tokenizer.note_attribute_order.index("offset/beat")] = 1
                    # x_a[:,model.tokenizer.note_attribute_order.index("onset/beat")] = 1

                    # for all attributes, set undefined to possible
                    for attribute in model.tokenizer.note_attribute_order:
                        x_a[:,model.tokenizer.note_attribute_order.index(attribute), model.tokenizer.token2idx[attribute + ":-"]] = 1

                    mask = einops.rearrange(x_a,"n_notes n_attributes vocab -> (n_notes n_attributes) vocab")[None,...]

                elif task == "pitch_onset_offset_set":
                    top_p = TOPP
                    enforce_prior = True

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
                
                elif task == "variation":
                    top_p = TOPP
                    prior_strength = 0.75
                    enforce_prior = False

                    x1h = torch.nn.functional.one_hot(batch[sample_idx], num_classes=len(model.tokenizer.vocab)).float()
                    mask = x1h[None, ...]

                elif task == "replace_bass":
                    top_p = TOPP
                    enforce_prior = True

                    x = batch[sample_idx]
                    mask = model.tokenizer.replace_instruments_mask(
                        x,
                        ["Bass"],
                        ["Bass"],
                        10,
                        275
                    )[None, ...].float()

                elif task == "replace_chromatic":
                    top_p = TOPP
                    enforce_prior = True

                    x = batch[sample_idx]
                    mask = model.tokenizer.replace_instruments_mask(
                        x,
                        ["Piano","Guitar"],
                        ["Chromatic Percussion"],
                        10,
                        275
                    )[None, ...].float()

                elif task == "replace_drums":
                    top_p = TOPP
                    enforce_prior = True

                    x = batch[sample_idx]
                    mask = model.tokenizer.replace_instruments_mask(
                        x,
                        ["Drums"],
                        ["Drums"],
                        10,
                        275
                    )[None, ...].float()
                    
                

                masks.append(mask)

               

            mask = torch.cat(masks, dim=0).to(device)

            y = model.sample2(
                prior=mask,
                nb_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                top_p=TOPP,
                prior_strength=PRIOR_STRENGTH,
                enforce_prior=ENFORCE_PRIOR,
                enforce_multiply=MULTIPLY_PRIOR if task != "variation" else False,
            )

            for sample_index in range(y.shape[0]):
                # decode
                print(f"n_notes {tokenizer.decode(y[sample_index]).note_num()}")
                y_sm = tokenizer.decode(y[sample_index])
                y_sm = sm_fix_overlap_notes(y_sm)

                preview_sm(y_sm)

            # get batch
            # export batch
            export_batch(y, model.tokenizer, OUTPUT_DIR + f"/{task}/hz_{hidden_size}_steps_{N_STEPS}_topp_{top_p}_prior_{prior_strength}_enforce_{enforce_prior}")
            export_batch(y, model.tokenizer, DEMO_DIR + f"/{task}/")
# %%


            

stack_sample = y[2]

y_sm = tokenizer.decode(stack_sample)


print(y_sm.tracks)

print(y_sm.note_num())

onsets = [note.start for note in y_sm.tracks[0].notes]
pitches = [note.pitch for note in y_sm.tracks[0].notes]
velocities = [note.velocity for note in y_sm.tracks[0].notes]

# plot scatter plot
plt.figure(figsize=(10,10))
sns.scatterplot(x=onsets,y=pitches, alpha=0.2)# hue=velocities)
plt.show()

preview_sm(y_sm)

# %%
