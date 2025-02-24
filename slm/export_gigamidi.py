from datasets import load_dataset
from tqdm import tqdm
import symusic
from util import preview_sm
import pandas as pd
import torch
from util import loop_sm, crop_sm
import os

# outpath 
OUTPATH = "../gigamidi"


split_shorthands = {
    "validation": "val",
    "test": "tst",
    "train": "trn"
}


for split in split_shorthands.keys():
    #%%
    dataset = load_dataset("Metacreation/GigaMIDI", split=split)

    os.makedirs(f"{OUTPATH}/{split}", exist_ok=True)

    for sample in tqdm(dataset, total=len(dataset)):
        try:
            midi = symusic.Score.from_midi(sample["music"])
            midi.dump_midi(f"{OUTPATH}/{split}/{sample['md5']}.mid")
        except:
            continue
