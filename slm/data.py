import torch
import glob
import symusic
import pandas as pd
import itertools
from tqdm import tqdm

class MidiDataset(torch.utils.data.Dataset):
    def __init__(self, midi_pattern, split_md5s, genre_file, cache_path, path_filter_fn, tokenizer):

        genre = [
            "pop",
            "rock",
            "italian%2cfrench%2cspanish",
            "classical",
            "alternative-indie",
            "romantic",
            "metal",
            "traditional",
            "renaissance",
            "country",
            "punk",
            "baroque",
            "dance-eletric",
            "jazz",
            "modern",
            "rnb-soul",
            "blues",
            "hip-hop-rap",
            "medley",
            "folk",
            "instrumental",
            "newage",
            "midi karaoke",
            "hits of the 2000s",
            "reggae-ska",
            "hits of 2011 2020",
            "hits of the 1980s",
            "latino",
            "musical%2cfilm%2ctv",
            "christian-gospel",
            "grunge",
            "hits of the 1970s",
            "world",
            "early_20th_century",
            "funk",
        ]

        self.tokenizer = tokenizer
        
        midi_paths = glob.glob(midi_pattern, recursive=True)
        # filter midi files by n_bars
        midi_paths = [x for x in midi_paths if path_filter_fn(x)]
        midi_records = [{"path": x, "md5": x.split("/")[-2]} for x in midi_paths]

        print(f"Found {len(midi_records)} midi files")

        assert len(midi_records) > 0

        split_md5s = open(split_md5s, "r").readlines()
        split_md5s = [x.strip() for x in split_md5s]
        split_md5s = set(split_md5s)
        assert len(split_md5s) > 0
        # filter midi_records by split_md5s
        midi_records = [x for x in midi_records if x["md5"] in split_md5s]

        print(f"Found {len(midi_records)} midi files in split")

        # read genre data
        genre_df = pd.read_json(genre_file, lines=True)
        genre_df["genre"] = genre_df["genre"].apply(lambda l: [item for sublist in l for item in sublist])
        genre_df = genre_df[genre_df["md5"].isin([x["md5"] for x in midi_records])]

        # # print genres that occur more than 10 times
        # genre_counts = genre_df["genre"].explode().value_counts()
        # genre_counts = genre_counts[genre_counts > 10]
        # print(f"Found {len(genre_counts)} genres that occur more than 10 times")
        # for genre, count in genre_counts.items():
        #     print(f"{genre}: {count}")
        

        print(f"Loaded {len(genre_df)} genre records")

        md5_to_genre = {x["md5"]: x["genre"] for x in genre_df.to_dict(orient="records")}

        print(f"Loading midi files...")

        midi_records = [{**x, "genre": md5_to_genre[x["md5"]] if x["md5"] in md5_to_genre else []} for x in midi_records]

        # # count records with genre
        # midi_records_with_genre = [x for x in midi_records if len(x["genre"]) > 0]
        # print(f"Found {len(midi_records_with_genre)} midi files with genre")

        # midi_records = [list(v) for k, v in itertools.groupby(midi_records, key=lambda x: x["md5"])]

        # # midi_records = midi_records

        # # print top 10 genres
        # genre_counts = {}
        # for record in midi_records:
        #     for genre in record[0]["genre"]:
        #         if genre not in genre_counts:
        #             genre_counts[genre] = 0
        #         genre_counts[genre] += 1
        # genre_counts = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        # print(f"Top 10 genres: {genre_counts[:100]}")

        # group by md5
        self.midi_records = [list(v) for k, v in itertools.groupby(midi_records, key=lambda x: x["md5"])]



    def __len__(self):
        return len(self.midi_records)
    
    def __getitem__(self, idx):
        return self.midi_records[idx]







        







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
