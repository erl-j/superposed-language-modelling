# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from slm.train_old import EncoderOnlyModel
from util import preview
import os
import IPython.display as ipd
from paper_checkpoints import checkpoints
import torch
import einops
import random

device = "cuda:3"
ROOT_DIR = "../"


mlm = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + checkpoints["mlm"],
        map_location=device,
    )
    .to(device)
    .eval()
)
mlm.enforce_constraint_in_forward = True


slm = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + checkpoints["slm"],
        map_location=device,
    )
    .to(device)
    .eval()
)


#%%
N_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path="../paper_assets/tst_midi_records_unique_pr.pt",
    path_filter_fn=lambda x: f"n_bars={N_BARS}" in x,
    genre_list=slm.tokenizer.config["tags"],
    tokenizer=slm.tokenizer,
    # min_notes=8 * N_BARS,
    min_notes=8*N_BARS,
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
SEED = 0

metrics = []
x_idx = next(iter(dl)).to(device)
# make copy

SUPERPOSITION_SCHEMES = ["random 0.25", "random 0.50", "random 0.75","random 0.90", "random 1.0", "tokens in masked", "tokens in whole sequence"]
MASK_LEVELS = ["position", "note"]

for superposition_scheme in SUPERPOSITION_SCHEMES:
    for mask_level in MASK_LEVELS:
        for masking_ratio in np.linspace(0.1, 0.9, 4):

            if mask_level == "position":
            
                n_masked_positions = int(n_positions * masking_ratio)    

                masked_position_idx = torch.tensor(random.sample(range(n_positions), n_masked_positions))

            elif mask_level == "note":
                n_masked_notes = int(n_notes * masking_ratio)

                masked_note_idx = torch.tensor(random.sample(range(n_notes), n_masked_notes))

                #include all indices for the attributes
                masked_position_idx = torch.tensor([idx + i * n_attributes for i in masked_note_idx for idx in range(n_attributes)])

            position_mask = torch.nn.functional.one_hot(masked_position_idx, n_positions).float().sum(dim=0).to(device)

            x = torch.nn.functional.one_hot(x_idx, vocab_size).float()

            if "random" in superposition_scheme:
                superposition_density = float(superposition_scheme.split(" ")[-1])

                superposition = torch.rand(x.shape).to(device) < superposition_density
            
            elif superposition_scheme == "tokens in masked":
                superposition = (position_mask[None,:, None] * x ).sum(dim=1, keepdim=True)

            elif superposition_scheme == "tokens in whole sequence":
                superposition = (x).sum(dim=1, keepdim=True)


            x_masked = x.clone() + superposition * position_mask[None,:, None]


            x_masked = x_masked.clamp(0, 1)

            x_masked = x_masked.to(device)

            for model in [mlm,slm]:
                # get the output

                logits = model(x_masked)

                # get the masked indices
                masked_logits = logits[:, masked_position_idx, :]
                masked_target_idx = x_idx[:, masked_position_idx]

                # get cross entropy loss
                loss = torch.nn.functional.cross_entropy(
                    masked_logits.reshape(-1, vocab_size), masked_target_idx.flatten()
                ).detach().cpu()

                top_k_accuracies = {}
                # compute top k accuracy
                for top_k in [1, 3, 5, 10, 20]:
                    top_k_accuracies[f"top_{top_k}_accuracy"] = torch.topk(
                        masked_logits, top_k, dim=-1
                    ).indices.eq(masked_target_idx.unsqueeze(-1)).any(dim=-1).float().mean().item()


                print(f"Masking Ratio: {masking_ratio:.2f}, Loss: {loss.item()}")

                model_name = ""
                if model.standard_mlm_forward:
                    model_name = "MLM"
                elif model.standard_mlm_forward == False:
                    model_name = "SLM"
                # elif model == mlm_no_rs:
                #     model_name = "MLM (No RS)"
                metrics.append(
                    {   
                        "superposition_scheme": superposition_scheme,
                        "mask_level": mask_level,
                        "masking_ratio": masking_ratio,
                        "loss": loss.item(),
                        "model": model_name,
                        **top_k_accuracies
                    }
                )

#%%
        
# go mask levels and plot for different superposition schemes
import pandas as pd

metrics = pd.DataFrame(metrics)

for mask_level in MASK_LEVELS:
    for superposition_scheme in SUPERPOSITION_SCHEMES:
        plt.figure(figsize=(10, 10))
        sns.lineplot(data=metrics[(metrics["mask_level"] == mask_level) & (metrics["superposition_scheme"] == superposition_scheme)], x="masking_ratio", y="top_3_accuracy", hue="model")
        plt.title(f"Mask Level: {mask_level}, Superposition Scheme: {superposition_scheme}")
        plt.show()
#%% 

x_idx = next(iter(dl)).to(device)

metrics = []
for scenario in ["standard"]:
    for attribute in ["pitch","onset/beat","offset/beat","instrument"]:

        attribute_index = mlm.tokenizer.note_attribute_order.index(attribute)

        masked_position_idx = torch.tensor([attribute_index + i * n_attributes for i in range(n_notes)])

        x = torch.nn.functional.one_hot(x_idx, vocab_size).float()

        x_masked = x.clone() 
        x_masked = einops.rearrange(x_masked, "b (n a) v -> b n a v", n=n_notes, a=n_attributes)
        x_beavr =  einops.rearrange(x, "b (n a) v -> b n a v", n=n_notes, a=n_attributes)
        if scenario == "easy":
            # to constraint
            x_masked[:,:, attribute_index] = x_masked[:,:, attribute_index].sum(dim=2, keepdim=True) > 0
        elif scenario == "standard":
            # mask the attribute
            x_masked[:, :, attribute_index,:] = 1
        elif scenario == "hard":
            # mask the attribute
            x_masked[:, :, attribute_index,:] = 1
            for attribute_index2 in range(n_attributes):
                if attribute_index2 != attribute_index:
                    x_masked[:, :, attribute_index2,:] = x_masked[:,:, attribute_index2].sum(dim=2, keepdim=True) > 0
        elif scenario == "0.5 random":
            noise = torch.rand(x_masked.shape).to(device) < 0.5
            for attribute_index2 in range(n_attributes):
                if attribute_index2 != attribute_index:
                    x_masked[:, :, attribute_index2,:] = torch.clamp(x_masked[:, :, attribute_index2,:] + noise[:, :, attribute_index2,:], 0, 1)


        else:
            raise ValueError("Invalid Scenario")


        # reshape
        x_masked = einops.rearrange(x_masked, "b n a v -> b (n a) v")

        print(f"Attribute: {attribute}")

        for model in [mlm, slm]:

            probs = model.compute_perplexity(x_masked[:2],x[:2])

            # logits = model(x_masked)

            # # get the masked indices
            # masked_logits = logits[:, masked_position_idx, :]
            # masked_target_idx = x_idx[:, masked_position_idx]

            # # get cross entropy loss
            # loss = torch.nn.functional.cross_entropy(
            #     masked_logits.reshape(-1, vocab_size), masked_target_idx.flatten()
            # ).detach().cpu()

            # top_k_accuracies = {}
            # # compute top k accuracy
            # for top_k in [1, 3, 5, 10, 20]:
            #     top_k_accuracies[f"top_{top_k}_accuracy"] = torch.topk(
            #         masked_logits, top_k, dim=-1
            #     ).indices.eq(masked_target_idx.unsqueeze(-1)).any(dim=-1).float().mean().item()

            # print(f"Attribute: {attribute}, Loss: {loss.item()}")

        
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
                    # "loss": loss.item(),
                    # **top_k_accuracies,
                    "log_probs": probs,
                }
            )

            print(metrics[-1])

#%%

import pandas as pd
metrics = pd.DataFrame(metrics)

for scenario in ["standard"]:
    # get means of log probs
    metrics["log_probs"] = metrics["log_probs"].apply(lambda x: x.mean().item())

    plt.figure(figsize=(10, 10))
    sns.barplot(data=metrics[metrics["scenario"] == scenario], x="attribute", y="log_probs", hue="model")
    plt.title(f"Scenario: {scenario}")
    plt.show()
#%%
        


# number of masked notes

# lines
# mlm (no rs)
# instrument set is known
# velocity set is known
# pitch set is known
# onset beat is known

# instrument values known
# velocity values known
# pitch values known
# onset beat values known


#%%







        





# if drum and confounding values are not drum pitches, we get ..

# effective 


# However, this since we have some structure we do the following.

# We constrain it to the set of pitches present.
        
# Constrain it to the insturments
        
# Constrain it to velocities
        
# Constrain it to the time steps
        
# Constrain it to tag
        
# update code to deal with B, E, A, V, Representation.
        

        






        


       

## structure aware.
        
    


for known_token_ratio in np.linspace(0, 1, 5):
    # get perforrance curve
    metrics = slm.performance_curve(val_dl, known_token_ratio=known_token_ratio)

    
    # plot metrics
    plt.plot(metrics)
    plt.title(f"Known Token Ratio: {known_token_ratio}")
    plt.show()



batch = next(iter(val_dl)).to(device)

masking_ratios = np.linspace(0.1, 1, 5)

superposition_density = np.linspace(0.1, 1, 5)

seed = 0

# set seed
torch.manual_seed(seed)

losses = []

max_notes = model.tokenizer.config["max_notes"]
n_attributes = len(model.tokenizer.note_attribute_order)

seq_len = max_notes * n_attributes


for masking_ratio in masking_ratios:

    x = batch.clone() 
    x = torch.nn.functional.one_hot(x, len(model.tokenizer.vocab)).float()

    x_target_idx = torch.argmax(x, dim=-1)
    
    # create mask for the input
    mask_noteattr_idx = random.sample( 
        range(seq_len),
        k= int(seq_len * masking_ratio)
    )

    x_masked = x

    x_masked[:, mask_noteattr_idx, :] = 1
   
    # get the output
    with torch.no_grad():
        logits = model(x_masked)

        # get masked indices

        masked_logits = logits[:, mask_noteattr_idx, :]
        masked_target_idx = x_target_idx[:, mask_noteattr_idx]


        # get cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            masked_logits.reshape(-1, len(model.tokenizer.vocab)),
            masked_target_idx.flatten()
        )

        losses.append({
            "masking_ratio": masking_ratio,
            "loss": loss.item()
        })

    print(f"Masking Ratio: {masking_ratio:.2f}, Loss: {loss.item()}")

#%%
# plot masking ratio on x and loss on y
import pandas as pd

losses = pd.DataFrame(losses)

plt.plot(losses["masking_ratio"], losses["loss"])
plt.xlabel("Masking Ratio")
plt.ylabel("Cross Entropy Loss")
plt.title("Masking Ratio vs Loss")

#%%




        





        

    

        



    # get the output
    




    



#%%
pos_z = model.positional_encoding[0].detach().cpu()[:model.tokenizer.config["max_notes"]]

print(pos_z.shape)

plt.figure(figsize=(10, 10))
sns.heatmap(pos_z.T, cmap="viridis")
plt.title("Positional Encoding")
plt.show()


#%%


#%%




# get one batch
x = next(iter(val_dl))

# move to device
x = x.to(device)

for base_masking_ratio in np.linspace(0, 1, 5):
    # get perofmrance curve
    metrics = model.performance_curve(x, base_masking_ratio=base_masking_ratio)
    # plot metrics
    plt.plot(metrics)
    plt.title(f"Base Masking Ratio: {base_masking_ratio}")
    plt.show()


# %%


# %%


# %%




# %%
