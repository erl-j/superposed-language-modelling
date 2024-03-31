# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import piano_roll
import os
import IPython.display as ipd
from util import get_scale
from paper_checkpoints import SLM_CKPT_PTH, MLM_CKPT_PTH, SS_SLM_CKPT_PTH
import torch
import random
import einops


# %%
device = "cuda:7"
ROOT_DIR = "../"

slm = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + SS_SLM_CKPT_PTH,
        map_location=device,
    )
    .to(device)
    .eval()
)

mlm = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + MLM_CKPT_PTH,
        map_location=device,
    )
    .to(device)
    .eval()
)


print(slm.enforce_constraint_in_forward)
mlm.mlm_restricted_sampling = True

#%%

N_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path="../artefacts/tst_midi_records.pt",
    path_filter_fn=lambda x: f"n_bars={N_BARS}" in x,
    genre_list=slm.tokenizer.config["tags"],
    tokenizer=slm.tokenizer,
    # min_notes=8 * N_BARS,
    min_notes=8 * N_BARS,
    max_notes=slm.tokenizer.config["max_notes"],
)
BATCH_SIZE = 64
# get val dataloader
dl = torch.utils.data.DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
# we want to see how much the predictions improve as we
# increase the information we get from neighbours

n_notes = mlm.tokenizer.config["max_notes"]
n_attributes = len(mlm.tokenizer.note_attribute_order)
vocab_size = len(mlm.tokenizer.vocab)
n_positions = n_notes * n_attributes

format_mask = slm.format_mask

event_format_mask = format_mask[: len(mlm.tokenizer.note_attribute_order)]

# for each attribute, find the indices of the tokens which belong to that attribute
attr_token_indices = [
    torch.where(event_format_mask[:, i])[0] for i in range(n_attributes)
]

#%% 

x_idx = next(iter(dl)).to(device)

metrics = []
for scenario in ["easy", "standard","hard"]:
    for attribute in mlm.tokenizer.note_attribute_order:

        attribute_index = mlm.tokenizer.note_attribute_order.index(attribute)

        masked_position_idx = torch.tensor([attribute_index + i * n_attributes for i in range(n_notes)])

        x = torch.nn.functional.one_hot(x_idx, vocab_size).float()

        x_masked = x.clone() 
        x_masked = einops.rearrange(x_masked, "b (n a) v -> b n a v", n=n_notes, a=n_attributes)


        if scenario == "easy":
            # to constraint
            x_masked[:,:, attribute_index] = x_masked[:,:, attribute_index].sum(dim=1, keepdim=True) > 0
        elif scenario == "standard":
            # mask the attribute
            x_masked[:, :, attribute_index,:] = 1
        elif scenario == "hard":
            # mask the attribute
            # x_masked[:, :, attribute_index,:] = 1
            for attribute_index2 in range(n_attributes):
                # if attribute_index2 != attribute_index:
                x_masked[:, :, attribute_index2,:] = x_masked[:,:, attribute_index2].sum(dim=1, keepdim=True) > 0
        else:
            raise ValueError("Invalid Scenario")


        # reshape
        x_masked = einops.rearrange(x_masked, "b n a v -> b (n a) v")

        mask_ratio = 0.5
        n_unmasked = int((1 - mask_ratio) * n_positions)
        x_masked[:, n_unmasked:] = x[:, n_unmasked:]

        x_masked = x_masked * slm.format_mask[None, :, :].to(x_masked.device)

        # plot mask
        # plt.figure(figsize=(10, 10))
        # plt.imshow(x_masked[0,:9*10].T.cpu().numpy(), aspect="auto")
        # plt.title(f"Attribute: {attribute}, Scenario: {scenario}")


        print(f"Attribute: {attribute}")

        for model in [mlm, slm]:

            logits = model(x_masked)

            # get the masked indices
            masked_logits = logits[:, masked_position_idx, :]
            masked_target_idx = x_idx[:, masked_position_idx]

            # # get cross entropy loss
            loss = torch.nn.functional.cross_entropy(
                masked_logits.reshape(-1, vocab_size), masked_target_idx.flatten()
            ).detach().cpu()

            top_k_accuracies = {}
            # compute top k accuracy
            for top_k in [1, 3, 5, 10, 20]:
                top_k_accuracies[f"top_{top_k}_accuracy"] = torch.topk(
                    masked_logits, top_k, dim=-1
                ).indices.eq(masked_target_idx.unsqueeze(-1)).any(dim=-1).float().mean().item()

            print(f"Attribute: {attribute}, Loss: {loss.item()}")

        
            model_name = ""
            if model.standard_mlm_forward:
                model_name = "MLM"
            elif not model.standard_mlm_forward:
                model_name = "SLM"
            # elif model == mlm_no_rs:
            #     model_name = "MLM (No RS)"
            metrics.append(
                {
                    "attribute": attribute,
                    "model": model_name,
                    "scenario": scenario,
                    "loss": loss.item(),
                    **top_k_accuracies,
                    # "log_probs": probs,
                }
            )

            print(metrics[-1])

#%%

import pandas as pd
metrics = pd.DataFrame(metrics)

for scenario in ["easy", "standard","hard"]:
    # get means of log probs
    # metrics["log_probs"] = metrics["log_probs"].apply(lambda x: x.mean().item())

    plt.figure(figsize=(10, 10))
    sns.lineplot(data=metrics[metrics["scenario"] == scenario], x="attribute", y="top_10_accuracy", hue="model")
    plt.title(f"Scenario: {scenario}")
    plt.show()
# %%
