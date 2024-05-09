#%%
from dirichlet_flow import DirichletFlowModel
import torch
from data import MidiDataset
import copy
device = 'cuda:0'

# ckpt = "../checkpoints/zany-thunder-48/last.ckpt"
ckpt = "../checkpoints/leafy-wood-51/last.ckpt"
model = DirichletFlowModel.load_from_checkpoint(
    ckpt, device=device
).to(device)


dataset = "mmd_loops"


ROOT_DIR = "../"
TMP_DIR = ROOT_DIR + "artefacts/tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/output"

MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR+f"data/{dataset}/tst_midi_records_unique_pr.pt",
    path_filter_fn=lambda x: f"n_bars={MODEL_BARS}" in x if dataset=="mmd_loops" else True,
    genre_list=model.tokenizer.config["tags"],
    tokenizer=model.tokenizer,
    min_notes=8 * MODEL_BARS,
    max_notes=model.tokenizer.config["max_notes"],
)

RESAMPLE_IDX = 1400

x = ds[RESAMPLE_IDX]
x_sm = model.tokenizer.decode(x)

#%%


mask = model.tokenizer.infilling_mask(
    x=x,
    beat_range=(4, 12),
    min_notes=0,
    max_notes=290,
)

mask = torch.tensor(mask * model.tokenizer.get_format_mask()).float()

prior = mask / mask.sum(dim=-1, keepdim=True)[None,:]


sampling_args = copy.deepcopy(model.flow_args)
sampling_args.num_integration_steps = 100
sampling_args.flow_temp = 0.5
sampling_args.alpha_spacing = 0.02
sampling_args.alpha_max = 8.0
# sampling_args.alpha_max = 30.0
# sampling_args.alpha_scale = 2.0
# sampling_args.alpha_max = 30.0
# sampling_args.alpha_scale = 2.0

l,y = model.sample(
    prior=None,
    sampling_args=sampling_args,
    break_on_anomaly=True,
    log= True,
)

y = y.argmax(dim=-1)

y_sm = model.tokenizer.decode(y[0].cpu().numpy())


from util import preview_sm

preview_sm(y_sm)



# %%
