import torch
import glob
import symusic
import pandas as pd
import itertools
from tqdm import tqdm


midi_pattern="../loop-detection/loops/annot_joblib/**/*.mid"
genre_file="../midi_data/metamidi_dataset/MMD_scraped_genre.jsonl"


for split in ["trn", "val", "tst"]:
    split_md5s=f"./split/{split}_md5s.txt"

    midi_paths = glob.glob(midi_pattern, recursive=True)

    # midi_paths = midi_paths[:1000]
    # filter midi files by n_bars
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

    # print genres that occur more than 10 times
    genre_counts = genre_df["genre"].explode().value_counts()
    genre_counts = genre_counts[genre_counts > 10]
    print(f"Found {len(genre_counts)} genres that occur more than 10 times")
    for genre, count in genre_counts.items():
        print(f"{genre}: {count}")


    print(f"Loaded {len(genre_df)} genre records")

    md5_to_genre = {x["md5"]: x["genre"] for x in genre_df.to_dict(orient="records")}

    print(f"Loading midi files...")

    midi_records = [{**x, "genre": md5_to_genre[x["md5"]] if x["md5"] in md5_to_genre else []} for x in tqdm(midi_records)]

    # load midi files with symusic
    midi_records = [{**x, "midi": symusic.Score(x["path"])} for x in tqdm(midi_records)]

    midi_records = [list(v) for k, v in itertools.groupby(midi_records, key=lambda x: x["md5"])]

    torch.save(midi_records, f"./artefacts/{split}_midi_records.pt")