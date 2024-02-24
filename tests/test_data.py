from slm import data

def test_MidiDataset():

    ds = data.MidiDataset(
        midi_pattern="../loop-detection/loops/annot_joblib/**/*.mid",
        split_md5s="./split/trn_md5s.txt",
        genre_file="../midi_data/metamidi_dataset/MMD_scraped_genre.jsonl",
    )

    assert len(ds) > 0