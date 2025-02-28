#%%
from symusic import Score, Synthesizer, BuiltInSF3 ,dump_wav
import glob
from tqdm import tqdm
import soundfile as sf
import os
import json

records_path = "../artefacts/audio/aba_records.jsonl"
scores_path = "../artefacts/audio/output.jsonl"
midi_path = "../artefacts/applications_250e/ground_truth/"

# load audio records and scores and merge
with open(records_path,"r") as f:
    records = [json.loads(line) for line in f.readlines()]

with open(scores_path,"r") as f:
    scores = [json.loads(line) for line in f.readlines()]

# merge records and scores by index
records = [{**record, **score} for record, score in zip(records, scores)]

# save audio records with scores with pandas
import pandas as pd
df = pd.DataFrame(records)
df.to_csv("../artefacts/audio/aba_records.csv")


# load audio with soundfile 
sample_rate = 44100

records = [
    {
        **record,
        "audio": sf.read("../artefacts/audio/" + record["path"])[0].T,
        "sample_rate": sample_rate
    }
    for record in tqdm(records)
]

# add midi to records
records = [
    {
        **record,
        "midi": Score(midi_path + record["path"].replace(".wav",".mid"))
    }
    for record in tqdm(records)
]

# 

# show distribution of "CE", "CU", "PC", "PQ"
import pandas as pd
import numpy as np

df = pd.DataFrame(records)

#%%
# plot histogram of "CE", "CU", "PC", "PQ"
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2)

for m in ["CE", "CU", "PC", "PQ"]:
    ax = axs.flatten()[["CE", "CU", "PC", "PQ"].index(m)]
    ax.hist(df[m], bins=20)
    ax.set_title(m)


#%%
import IPython.display as ipd
from IPython.display import Audio, display

for m in ["CE", "CU", "PC", "PQ"]:
    print(f"Top 3 in {m}")

    # play top 10 in content enjoyment
    top_3 = df.sort_values(m, ascending=False).head(3)
    for i, record in top_3.iterrows():
        print(record["path"])
        display(Audio(record["audio"], rate=44100))

    print(f"Worst 3 in {m}")
    # get five worst in content enjoyment
    worst_3 = df.sort_values(m, ascending=True).head(3)

    for i, record in worst_3.iterrows():
        print(record["path"])
        display(Audio(record["audio"], rate=44100))


#%%


# %%


