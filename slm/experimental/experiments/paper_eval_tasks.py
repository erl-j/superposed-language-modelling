import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from slm.train_old import EncoderOnlyModel
from util import piano_roll
import os
import IPython.display as ipd
from util import get_scale
from paper_checkpoints import SLM_CKPT_PTH, MLM_CKPT_PTH
import torch
import random

device = "cuda:7"
ROOT_DIR = "../"

slm = (
    EncoderOnlyModel.load_from_checkpoint(
        ROOT_DIR + SLM_CKPT_PTH,
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

#%%

