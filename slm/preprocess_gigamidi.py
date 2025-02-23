#%%
from datasets import load_dataset
from tqdm import tqdm
import symusic
from util import preview_sm
import pandas as pd
import torch
from util import loop_sm, crop_sm

#%%
TPQ = 96
LIMIT = 1_000_000_000

ds_path = "./data/gmd_loops/"


genre_set = open(f"{ds_path}/tags.txt").read().splitlines()

split_shorthands = {
    "train": "trn",
    "validation": "val",
    "test": "tst"
}

for split in split_shorthands.keys():
    #%%
    dataset = load_dataset("Metacreation/GigaMIDI", split=split)

    #%% 

    # get length of dataset
    ds_len = len(dataset)

    print(f"Dataset length: {ds_len} pieces")

    records = []
    for sample in tqdm(dataset, total=len(dataset)):
        try:
            midi = symusic.Score.from_midi(sample["music"])
            records.append(
                {**sample, "midi": midi}
            )
        except:
            continue

    print(f"Pieces that loaded: {len(records)}")

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
    TARGET_TPQ = 96
    records = [{**record, "midi": record["midi"].resample(TARGET_TPQ, min_dur=0)} for record in tqdm(records)]
    #%%

    #%%
    print("Filter out pieces that have less than 4 notes")
    records = [record for record in records if record["midi"].note_num() >= 4]
    print(f"Pieces remaining: {len(records)}")
    #%%
    def get_n_bars(score, tpq, quarters_per_bar):
        return score.end() / (tpq * quarters_per_bar)

    print("Get number of bars")

    records = [{**record, "n_bars": get_n_bars(record["midi"], TARGET_TPQ, 4)} for record in records]

    #%%
    # plot stats of number of bars
    pd.Series([record["n_bars"] for record in records]).value_counts()
    print(f"Min number of bars: {min([record['n_bars'] for record in records])}")
    print(f"Max number of bars: {max([record['n_bars'] for record in records])}")
    print(f"Mean number of bars: {sum([record['n_bars'] for record in records]) / len(records)}")
    print(f"Median number of bars: {sorted([record['n_bars'] for record in records])[len(records) // 2]}")

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

    # if instrument_category is "drums-only", set all tracks to drums
    print("Set drum only tracks to really only have drums")
    records = [
        {**record, "midi": set_to_drums(record["midi"])} if record["instrument_category"] == "drums-only" else record
        for record in records ]

    #%%
    # make histogram of number of bars
    pd.Series([record["n_bars"] for record in records]).hist(bins=10, range=(0, 32))


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

    print(f"Number of pieces: {len(records)}")
    print("Filter out expressive solo piano")
    records = [record for record in records if not is_expressive_solo_piano(record)]
    print(f"Pieces remaining: {len(records)}")
        
        
    #%%

    def extract_loops(sample, target_bars):

        # first, estimate the intended length of the loop
        # we round to the nearest bar
        loops = []
        estimated_bars = round(sample["n_bars"])

        # print tpq
        # print(f"TPQ: {sample['midi'].tpq}")

        genres = record["genres_scraped"] if record["genres_scraped"] is not None else []
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
            loop_points = list(zip(sample["loops"]["start_tick"], sample["loops"]["end_tick"]))
            # print(f"len of loop points: {len(loop_points)}")
            # get unique loop points
            loop_points = list(set(loop_points))
            # print(f"len of unique loop points : {len(loop_points)}")
            for start_tick, end_tick in loop_points:
                # check if n_bars is less or equal to target_bars
                n_bars = (end_tick - start_tick) // (TARGET_TPQ * 4)
                # round
                # print(f"n_bars: {n_bars}")
                n_bars = round(n_bars)
                # print(f"n_bars: {n_bars}")
                if n_bars == 0:
                    continue
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

    import random
    random_idx = random.randint(0, len(records))
    record = records[random_idx]

    # preview sm
    preview_sm(record["midi"])
    # print record keys and items for thigs where items are strings
    for key, item in record.items():
        if isinstance(item, str):
            print(f"{key}: {item}")
    # test to extract loops from first record
    loops = extract_loops(record, 4)
    print(f"Number of loops: {len(loops)}")

    # preview the loops
    for loop in loops:
        looped = loop_sm(loop["midi"], 4, 2)
        preview_sm(looped)

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
    torch.save(loop_records, f"{ds_path}/{split_shorthands[split]}_midi_records.pt")