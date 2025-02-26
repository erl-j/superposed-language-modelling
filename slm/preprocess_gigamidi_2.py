#%%
from datasets import load_dataset
from tqdm import tqdm
import symusic
from util import preview_sm
import pandas as pd
import torch
from util import loop_sm, crop_sm

# TODO: CHECK CONSISTENCY BETWEEN TRACKS AND INSTRUMENT CATEGORY BETTER
# TODO: MAKE SURE THAT ALL LOOPS HAVE TEMPOS. CLIP, SHIFT AND CROP might be messing with tempos

#%%
LIMIT = 1_000_000_000
TARGET_TPQ = 96

ds_path = "./data/gmd_loops_2"

loop_df = pd.read_parquet(f"./data/loop_detection_results.parquet")
loop_df["md5"] = loop_df["midi_file_path"].apply(lambda x: x.split("/")[-1].replace(".mid", ""))
# take unique by md5
loop_df = loop_df.drop_duplicates(subset="md5")

genre_set = open(f"{ds_path}/tags.txt").read().splitlines()

split_shorthands = {
    "validation": "val",
    "test": "tst",
    "train": "trn"
}

#%%
for split in ["validation", "test", "train"]:
    print(f"Processing {split} split")

    dataset = load_dataset("Metacreation/GigaMIDI", split=split)

    #%%



    # get dataset as pandas dataframe
    dataset_df = dataset.to_pandas()

    # join on md5
    dataset_df = dataset_df.merge(loop_df, on="md5", how="left")

    # replace nan dataset df with empty list
    dataset_df["candidate_loops"] = dataset_df["candidate_loops"].apply(lambda x: [] if isinstance(x, float) and pd.isna(x) else x)
    # Now verify the change
    print(f"Number of pieces with loops: {len(dataset_df[dataset_df['candidate_loops'].apply(len) > 0])}")

    #%%


    # get length of dataset
    ds_len = len(dataset_df)

    print(f"Dataset length: {ds_len} pieces")

    records = []
    for sample in tqdm(dataset_df.iterrows(), total=ds_len):
        sample = sample[1]
        try:
            midi = symusic.Score.from_midi(sample["music"])
            records.append(
                {**sample, "midi": midi}
            )
        except:
            continue

    print(f"Pieces that loaded: {len(records)}")

    records_with_candidate_loops = [record for record in records if len(record["candidate_loops"]) > 0]

    print(f"Number of records with candidate loops: {len(records_with_candidate_loops)}")


    #%%
    print("Filter out pieces that don't have tempo")
    # now exclude pieces that have no tempo
    records = [record for record in records if len(record["midi"].tempos) > 0]
    print(f"Pieces remaining: {len(records)}")

    #%% filter out pieces that don't have time signature
    print("Filter out pieces that don't have time signature")
    records = [record for record in records if len(record["midi"].time_signatures) > 0]
    print(f"Pieces remaining: {len(records)}")

    print("Filter out pieces that are not in 4/4")
    # now exclude pieces that are not in 4/4
    records = [record for record in records if record["midi"].time_signatures[-1].numerator == 4 and record["midi"].time_signatures[-1].denominator == 4]
    print(f"Pieces remaining: {len(records)}")

    #%%

    #%%
    # convert to tpq
    # print(f"Resample to {TARGET_TPQ} TPQ")
    # records = [{**record, "midi": record["midi"].resample(TARGET_TPQ, min_dur=0)} for record in tqdm(records)]
    #%%

    #%%
    print("Filter out pieces that have less than 4 notes")
    records = [record for record in records if record["midi"].note_num() >= 4]
    print(f"Pieces remaining: {len(records)}")
    #%%
    def get_n_bars(score, quarters_per_bar):
        return score.end() / (score.tpq * quarters_per_bar)

    print("Get number of bars")

    records = [{**record, "n_bars": get_n_bars(record["midi"], 4)} for record in records]

    #%%
    # plot stats of number of bars
    pd.Series([record["n_bars"] for record in records]).value_counts()
    print(f"Min number of bars: {min([record['n_bars'] for record in records])}")
    print(f"Max number of bars: {max([record['n_bars'] for record in records])}")
    print(f"Mean number of bars: {sum([record['n_bars'] for record in records]) / len(records)}")
    print(f"Median number of bars: {sorted([record['n_bars'] for record in records])[len(records) // 2]}")

    #%% plot histogram of number of bars
    pd.Series([record["n_bars"] for record in records]).hist(bins=10, range=(0, 32))

    #%% 
    # what do 4 and 8 bars look like
    four_bars_records = [record for record in records if record["n_bars"] <= 4]
    print(f"Number of 4 bars or less: {len(four_bars_records)}")
    # preview 5 
    for record in four_bars_records[:5]:
        preview_sm(record["midi"])

    # how many instruments do they have on average
    print(f"Average number of instruments: {sum([len(record['midi'].tracks) for record in four_bars_records]) / len(four_bars_records)}")

    #%%
    eight_bars_records = [record for record in records if record["n_bars"] <= 8]

    print(f"Number of 8 bars or less: {len(eight_bars_records)}")
    # preview 5
    for record in eight_bars_records[:5]:
        preview_sm(record["midi"])

    # how many instruments do they have on average
    print(f"Average number of instruments: {sum([len(record['midi'].tracks) for record in eight_bars_records]) / len(eight_bars_records)}")

    #%%
    # filter out pieces that have less than 1 bar
    print("Filter out pieces that have less than 0.75 bar")
    records = [record for record in records if record["n_bars"] >= 0.75]
    print(f"Pieces remaining: {len(records)}")

    #%% set drum only tracks to really only have drums
    def set_to_drums(score):
        for track_idx in range(len(score.tracks)):
            score.tracks[track_idx].is_drum = True
            score.tracks[track_idx].program = 0
        return score

    def has_drums(score):
        for track in score.tracks:
            if track.is_drum and track.note_num() > 0:
                return True
        return False

    def has_non_drum(score):
        for track in score.tracks:
            if not track.is_drum and track.note_num() > 0:
                return True
        return

    # check how many "drums-only" have non drums

    print("Check how many 'drums-only' have non drums")
    records_with_non_drum = [record for record in records if has_non_drum(record["midi"]) and record["instrument_category"] == "drums-only"]
    print(f"Number of 'drums-only' with non drums: {len(records_with_non_drum)}")

    # check how many "no-drums" have drums
    print("Check how many 'no-drums' have drums")
    records_with_drums = [record for record in records if has_drums(record["midi"]) and record["instrument_category"] == "no-drums"]
    print(f"Number of 'no-drums' with drums: {len(records_with_drums)}")

    # make sure all "drums-only" have drums

    # if instrument_category is "drums-only", set all tracks to drums
    print("Set drum only tracks to really only have drums")
    records = [
        {**record, "midi": set_to_drums(record["midi"])} if record["instrument_category"] == "drums-only" else record
        for record in records ]

    # filter out pieces that are "no-drums" but have drums
    print("Filter out pieces that are 'no-drums' but have drums")
    records = [record for record in records if not (record["instrument_category"] == "no-drums" and has_drums(record["midi"]))]

    print(f"Pieces remaining: {len(records)}")

    #%%

    # filter out expressive solo piano
    def is_expressive_solo_piano(sample):
        if sample["median_metric_depth"] is None:
            return False
        # check if instrument is piano
        if sample["instrument_category"] != "no-drums":
            return False
        if len(sample["median_metric_depth"]) == 1:
            if sample["median_metric_depth"][0] > 10:
                return True
        return False


    # check how many expressive solo piano
    print("Check how many expressive solo piano")

    records_with_expressive_solo_piano = [record for record in records if is_expressive_solo_piano(record)]

    #%% 
    # play some examples
    for record in records_with_expressive_solo_piano[:5]:
        preview_sm(record["midi"])

    #%%

    print(f"Number of expressive solo piano: {len(records_with_expressive_solo_piano)}")

    records = [record for record in records if not is_expressive_solo_piano(record)]

    print(f"Pieces remaining: {len(records)}")

    #%%

    # has drums outside of range
    def has_drums_outside_range(score, min_pitch=35, max_pitch=81):
        for track in score.tracks:
            if track.is_drum:
                for note in track.notes:
                    if note.pitch < min_pitch or note.pitch > max_pitch:
                        return True
        return False

    # check how many have drums outside of range
    print("Check how many have drums outside of range")
    records_with_drums_outside_range = [record for record in records if has_drums_outside_range(record["midi"])]

    print(f"Number of pieces with drums outside of range: {len(records_with_drums_outside_range)}")
    # remove
    records = [record for record in records if not has_drums_outside_range(record["midi"])]

    print(f"Pieces remaining: {len(records)}")
    #%%
    # make histogram of number of bars
    pd.Series([record["n_bars"] for record in records]).hist(bins=10, range=(0, 32))

    #%%

    # save records
    full_records = []
    for record in records:
        genres = record["genres_scraped"] if record["genres_scraped"] is not None else []
        # keep only genres in genre_list
        genres = [genre for genre in genres if genre in genre_set]
        # make sure genres are unique
        genres = list(set(genres))
        full_records.append(
            {
                **record,
                "music": "",
                "midi": # resample to TARGET_TPQ
                    record["midi"].copy().resample(TARGET_TPQ, min_dur=0),
                "path": record["md5"],
                "md5": record["md5"],
                "genre": genres
            }
        )

    # save
    torch.save(full_records, f"{ds_path}/{split_shorthands[split]}_midi_records_full.pt")


    #%%
    print(records[0])
    #%%

    def extract_loops(sample, target_bars):

        # first, estimate the intended length of the loop
        # we round to the nearest bar
        loops = []
        estimated_bars = round(sample["n_bars"])

        print(f"Estimated bars: {estimated_bars}")

        # print tpq
        # print(f"TPQ: {sample['midi'].tpq}")

        genres = sample["genres_scraped"] if sample["genres_scraped"] is not None else []
        # keep only genres in genre_list
        genres = [genre for genre in genres if genre in genre_set]
        # make sure genres are unique
        genres = list(set(genres))

        if estimated_bars <= target_bars:
            # if estimated bars divides target bars, we can use the loop, we crop it and tile it
            if target_bars % estimated_bars == 0:
                crop_sm(sample["midi"], estimated_bars, 4)
                loop_midi = loop_sm(sample["midi"], estimated_bars, target_bars // estimated_bars)
                loops.append(loop_midi)
        else:
            # now we look for loops

            for loop in sample["candidate_loops"]:
                # check if n_bars is less or equal to target_bars
                n_bars =  int(loop["n_bars"])
                start_tick = loop["start_tick"]
                end_tick = loop["end_tick"]

                # round
                # print(f"n_bars: {n_bars}")
                # print(f"n_bars: {n_bars}")
                if n_bars <= target_bars:
                    # if n_bars divides target_bars, we can use the loop, we crop it and tile it
                    if target_bars % n_bars == 0:
                        loop_midi = sample["midi"].clip(
                            start=start_tick,
                            end=end_tick,
                            clip_end=True
                        ).shift_time(-start_tick)
                        # crop
                        loop_midi = crop_sm(loop_midi, n_bars, 4)
                        loop_midi = loop_sm(loop_midi, n_bars, target_bars // n_bars)
                        loops.append(loop_midi)
        # filter out loops that have less than 2 notes
        loops = [loop for loop in loops if loop.note_num() >= 2]
        loop_records = []
        for loop in loops:
            loop_records.append(
                {   
                    **sample,
                    "music": "",
                    "path": f"{sample['md5']}_n_bars={target_bars}",
                    "md5": sample["md5"],
                    "midi": loop,
                    "genre": genres
                }
            )
        return loop_records

    # # records with candidate loops
    # records_with_candidate_loops = [record for record in records if len(record["candidate_loops"]) > 0]

    # print(f"Number of records with candidate loops: {len(records_with_candidate_loops)}")

    # import random
    # random_idx = random.randint(0, len(records_with_candidate_loops) - 1)
    # record = records_with_candidate_loops[random_idx]

    # # preview sm
    # preview_sm(record["midi"])
    # # print record keys and items for thigs where items are strings
    # for key, item in record.items():
    #     if isinstance(item, str):
    #         print(f"{key}: {item}")
    # # test to extract loops from first record
    # loops = extract_loops(record, 4)
    # print(f"Number of loops: {len(loops)}")

    # # preview the loops
    # for loop in loops:
    #     print(f"genre: {loop['genre']}")
    #     looped = loop_sm(loop["midi"], 4, 2)
    #     # resample
    #     looped = looped.resample(24, min_dur=0)
    #     preview_sm(looped)


    #%%

    records_with_loops = [{**record, "extracted_loops": extract_loops(record, 4)} for record in tqdm(records)]


    #%%
    # now make a big list of all loops
    loop_records = [record["extracted_loops"] for record in records_with_loops]

    # 
    # flatten
    flat_loop_records = [record for records in loop_records for record in records]
    print(f"Number of loops: {len(flat_loop_records)}")

    #%% 
    # get statitics of instrument category in loops
    instrument_categories = [record["instrument_category"] for record in flat_loop_records]
    print(pd.Series(instrument_categories).value_counts())

    # save 
    torch.save(loop_records, f"{ds_path}/{split_shorthands[split]}_midi_records_loops.pt")