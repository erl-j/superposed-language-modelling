import torch
import glob
import symusic
import pandas as pd
import itertools
from tqdm import tqdm
import random
from augmentation import transpose_sm

class MidiDataset(torch.utils.data.Dataset):
    def __init__(self, cache_path, genre_list, path_filter_fn, tokenizer, transposition_range=[-7, 7]):
        self.tokenizer = tokenizer
        self.records = torch.load(cache_path)
        for i in range(len(self.records)):
            self.records[i] = [x for x in self.records[i] if path_filter_fn(x["path"])]
            self.records[i] = [{**x, "genre": [g for g in x["genre"] if g in genre_list]} for x in self.records[i]]
        self.records = [x for x in self.records if len(x) > 0]
        self.transposition_range = transposition_range

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = random.choice(self.records[idx])
        midi = record["midi"]
        if self.transposition_range is not None:
            transposition = random.randint(*self.transposition_range)
            midi = transpose_sm(midi, transposition)
        return torch.tensor(
            self.tokenizer.encode(midi
            ,random.choice(record["genre"] if len(record["genre"]) > 0 else ["other"])
            )
        )







        







# augmentation_config = {
#     "augmentation_transposition_range":[-7, 7],
#     "augmentation_tempo_scale": 0.03,
#     "augmentation_velocity_shift": 3,
# }
# tokenizer_config = {
#     "ticks_per_beat":24,
#     "pitch_range":[31, 108],
#     "max_beats":33,
#     "max_notes":128,
#     "min_tempo":50,
#     "max_tempo":200,
#     "n_tempo_bins": 16,
#     "time_signatures": None,
#     "tags": ["rock","pop"],
#     "shuffle_notes": True,
#     "use_offset": True,
#     "merge_pitch_and_beat":True,
#     "use_program": False,
# }
