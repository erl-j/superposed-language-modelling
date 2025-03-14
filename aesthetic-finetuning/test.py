
#%%
from audiobox_aesthetics.infer import initialize_predictor
predictor = initialize_predictor()



#%%
import symusic
from symusic import Synthesizer, BuiltInSF3
from tqdm import tqdm
import glob
import torch
import IPython.display as ipd
import os
def play_audio(audio):
    return ipd.display(ipd.Audio(audio, rate=44100))

sm = symusic.Score()

midi_path =  "../artefacts/constrained_generation/ground_truth/piano"

midi_paths = glob.glob(midi_path + "/*.mid")

import random

midi_paths = random.sample(midi_paths, 100)


render_path = "artefacts/fs_renders"

# And the following one is the default soundfont if you don't specify it when creating a synthesizer
sf_path = BuiltInSF3.MuseScoreGeneral().path(download=True)

# # sf3 and sf2 are both supported
# sf_path = "path/to/your/soundfont.sf3"

sample_rate = 44100

synth = Synthesizer(
    sf_path = sf_path, # the path to the soundfont
    sample_rate = sample_rate, # the sample rate of the output wave, sample_rate is the default value
)

#%%

records = []

for midi_path in tqdm(midi_paths):
    midi = symusic.Score(midi_path)
    audio = synth.render(midi, stereo=True) # stereo is True by default, which means you will get a stereo wave
    os.makedirs(render_path,exist_ok=True)
    # play_audio(audio)
    audio = torch.tensor(audio).float()
    scores = predictor.forward([{"path": audio, "sample_rate": sample_rate}])
    records.append(
        {"midi_path": midi_path, "audio": audio, **scores[0]}
    )

#%%
# create a dataframe to store the records
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.DataFrame(records)

# show "CE" distribution
sns.histplot(df["CE"], bins=20, binrange=(0,10))
plt.show()
sns.histplot(df["CU"], bins=20, binrange=(0,10))
plt.show()
sns.histplot(df["PQ"], bins=20, binrange=(0,10))
plt.show()
sns.histplot(df["PC"], bins=20, binrange=(0,10))
plt.show()

#%%

for attr in ["CE", "CU", "PQ", "PC"]:
    # play top 3 and bottom 3
    top3 = df.sort_values(attr, ascending=False).head(3)
    bottom3 = df.sort_values(attr, ascending=True).head(3)

    print(f"Top 3 {attr}")
    for i, row in top3.iterrows():
        print(row["midi_path"])
        play_audio(row["audio"])

    print(f"Bottom 3 {attr}")
    for i, row in bottom3.iterrows():
        print(row["midi_path"])
        play_audio(row["audio"])


    




# %%
