# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import piano_roll
import os
import IPython.display as ipd
from paper_checkpoints import SLM_CKPT_PTH, MLM_CKPT_PTH


device = "cuda:7"
ROOT_DIR = "../"

MODEL = "mlm"


model = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + SLM_CKPT_PTH if MODEL == "slm" else ROOT_DIR + MLM_CKPT_PTH,
        map_location=device,
    )
    .to(device)
    .eval()
)

#%%

MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR+"artefacts/tst_midi_records.pt",
    path_filter_fn=lambda x: f"n_bars={MODEL_BARS}" in x,
    genre_list=model.tokenizer.config["tags"],
    tokenizer=model.tokenizer,
    min_notes=8 * MODEL_BARS,
    max_notes=model.tokenizer.config["max_notes"],
)

OUTPUT_DIR = ROOT_DIR + "artefacts/examples_4"
TMP_DIR = ROOT_DIR + "artefacts/tmp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preview(sm, tmp_dir):
    # SAMPLE_RATE = 44_100
    os.makedirs(tmp_dir, exist_ok=True)
    midi_path = tmp_dir + "/tmp.mid"
    audio_path = tmp_dir + "/output.wav"
    sm.dump_midi(midi_path)
    pr = piano_roll(sm)
    plt.figure(figsize=(10, 10))
    sns.heatmap(pr, cmap="magma")
    plt.show()

    os.system(f"fluidsynth {midi_path} -F {audio_path}")
    ipd.display(ipd.Audio(audio_path))

RESAMPLE_IDX = 17009

x = ds[RESAMPLE_IDX]
x_sm = model.tokenizer.decode(x)

preview(x_sm, TMP_DIR)

x_sm.dump_midi(OUTPUT_DIR + "/resample_original.mid")


#%%
import torch


dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

x = next(iter(dl)).to(device)
#%%


for masking_ratio in np.linspace(0.1, 0.9, 9):
    for topp in np.linspace(0.1, 0.9, 9):

        mask = torch.rand((x.shape[0], x.shape[1]),device=device) < masking_ratio

        x1h = torch.nn.functional.one_hot(x, num_classes=len(model.tokenizer.vocab)).float()
        x_masked = x1h + mask[...,None] * torch.ones_like(x1h) 
        # clamp
        x_masked = torch.clamp(x_masked, 0, 1)


        model.enforce_constraint_in_forward = True


        logits = model.forward(x_masked)


        from util import top_k_top_p_filtering
        import einops

        logits = einops.rearrange(logits, "b s v -> (b s) v")

        logits = top_k_top_p_filtering(logits, top_k=0, top_p=topp, filter_value=-float("Inf"))

        logits = einops.rearrange(logits, "(b s) v -> b s v", b=x.shape[0], s=x.shape[1])

        probs = torch.nn.functional.softmax(logits, dim=-1)

        probs *= mask[...,None]

        probs = probs>1e-10

        # plot logits
        plt.figure(figsize=(10, 10))
        sns.heatmap(probs[0,:9*4].detach().cpu().numpy().T, cmap="magma")
        plt.title(f"masking_ratio={masking_ratio:.2f}, topp={topp:.2f}")
        plt.show()


# %%
        
# what constraints can be inferred from context?
