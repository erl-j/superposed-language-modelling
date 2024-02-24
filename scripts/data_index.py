#%%
import glob
import pandas
import pydash

#%%
N_BARS = 4

midi_paths = glob.glob("../loop-detection/loops/annot_joblib/**/*.mid", recursive=True)
midi_paths = [x for x in midi_paths if f"n_bars={N_BARS}" in x]

records = [{"path": x, "md5": x.split("/")[-2]} for x in midi_paths]
midi_df = pandas.DataFrame(records)


#%%
genre_path = "../midi_data/metamidi_dataset/MMD_scraped_genre.jsonl"
genre_df = pandas.read_json(path_or_buf=genre_path, lines=True)

# print len
print(len(genre_df))

#%%


# flatten genre column
genre_df["genre"] = genre_df["genre"].apply(lambda x: pydash.flatten(x))

# merge midi_df and genre_df
df = midi_df.merge(genre_df, on="md5", how="left")

# replace genre NaN with empty list
df["genre"] = df["genre"].apply(lambda x: x if isinstance(x, list) else [])
#%%

# group by md5, and make lists of genres, and paths

grouped = df.groupby("md5").agg({"genre": "sum", "path": "sum"})

#%%
print(grouped.head())

# %%
