#%%
from datasets import load_dataset
from tqdm import tqdm
import symusic
from util import preview_sm
import pandas as pd
import torch
from util import loop_sm, preview_sm
from preprocess_gigamidi import preprocess_sample, get_loop_records
#%%
print(f"Hello, world!")

TPQ = 96
LIMIT = 1_000

ds_path = "../data/gmd_loops/"
print(f"Dataset path: {ds_path}")

genre_set = open(f"{ds_path}/tags.txt").read().splitlines()


dataset = load_dataset("Metacreation/GigaMIDI", split="validation")

records = []
sample_idx = 0
for sample in tqdm(dataset):
    sample_idx += 1
    records.append(preprocess_sample(sample))
    if sample_idx >= LIMIT:
        break
print(f"Number of records: {len(records)}")

# filter 4/4
records = [record for record in records if record["time_sig"] == "4/4"]

# get loop records
loop_records = []
for record in records:
    loop_records.append(get_loop_records(record))
# %%
