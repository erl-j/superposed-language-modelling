import data

if __name__ == "__main__":
    ds = data.MidiDataset(
        midi_pattern="../loop-detection/loops/annot_joblib/**/*.mid",
        split_md5s="./split/trn_md5s.txt",
        genre_file="../midi_data/metamidi_dataset/MMD_scraped_genre.jsonl",
        path_filter_fn = lambda x: "n_bars=4" in x
    )

    print(len(ds))