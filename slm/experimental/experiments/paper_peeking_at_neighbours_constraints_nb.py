#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from slm.train_old import EncoderOnlyModel
from util import piano_roll
import os
import IPython.display as ipd
from util import get_scale
from slm.PAPER_CHECKPOINTS import SLM_CKPT_PTH, MLM_CKPT_PTH

#%%
device = "cuda:7"
ROOT_DIR = "../"

slm = EncoderOnlyModel.load_from_checkpoint(
    ROOT_DIR + SLM_CKPT_PTH,
    map_location=device,
    avg_positional_encoding=True,
).to(device).eval()

mlm = EncoderOnlyModel.load_from_checkpoint(
    ROOT_DIR + MLM_CKPT_PTH,
    map_location=device,
    avg_positional_encoding=True,
).to(device).eval()
#%%

# if a future notes are constrained to a major scale, then the current note will probably also belong to the same scale

scales = ["C major", "C pentatonic", "G major"]

for scale in scales:
    for model in [slm, mlm]:

        scale_midi_pitch = get_scale(scale, [-1, 127])


        mask = (
            model.tokenizer.constraint_mask(
                instruments = ["Piano"],
                scale=scale,
                min_notes=290,
            )[None, ...]
            .to(model.device)
            .float()
        )

        # make first note undefined
        mask[:,:len(model.tokenizer.note_attribute_order),:] = 1

        # forward pass through the model

        logits = model.forward(
            mask,
        )

        probs = logits.softmax(dim=-1)[0, :, :].detach().cpu().numpy()

        # look at the logits

        first_note_probs = probs[:len(model.tokenizer.note_attribute_order), :]

        # get the 

        # select pitch frame
        pitch_frame = model.tokenizer.note_attribute_order.index("pitch")

        # get tokens starting with "pitch:"
        pitch_tokens = [t for t in model.tokenizer.vocab if t.startswith("pitch:")]
        pitch_token_idxs = [model.tokenizer.token2idx[t] for t in pitch_tokens]

        # # get pitch frames
        pitch_attribute_index = model.tokenizer.note_attribute_order.index("pitch")

        # get the probs of the pitch frame
        pitch_probs = first_note_probs[pitch_attribute_index, pitch_token_idxs]

        # scale token_idxs
        scale_tokens = [f"pitch:{p}" for p in scale_midi_pitch]
        scale_token_idxs = [model.tokenizer.token2idx[t] for t in scale_tokens]
        
        highlight_idx = scale_token_idxs
        # # plot the probs of the pitch frame

        plt.figure(figsize=(10, 20))
        bars = plt.barh(np.arange(len(pitch_probs)), pitch_probs)
        plt.yticks(np.arange(len(pitch_probs)), pitch_tokens)

        for idx in highlight_idx:
            plt.gca().add_patch(plt.Rectangle((0, pitch_token_idxs.index(idx)-0.5), 1, 1, fill=True, color='red', alpha=0.5))

            
        plt.yticks(np.arange(len(pitch_probs)), pitch_tokens)
        
        plt.title(f"Probs if neighbour notes are constrained to {scale}")
        plt.show()

        # same for instrument
        # instrument_frame = model.tokenizer.note_attribute_order.index("instrument")
        # instrument_tokens = [t for t in model.tokenizer.vocab if t.startswith("instrument:")]
        # instrument_token_idxs = [model.tokenizer.token2idx[t] for t in instrument_tokens]

        # instrument_probs = first_note_probs[instrument_frame, instrument_token_idxs]

        # plt.figure(figsize=(10, 20))
        # # make bars horizontal
        # plt.barh(np.arange(len(instrument_probs)), instrument_probs)
        # plt.yticks(np.arange(len(instrument_probs)), instrument_tokens)

        # plt.title(f"Probs if neighbour notes are constrained to {scale}")
# %%
