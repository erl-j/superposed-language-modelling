#%%
import muspy
import glob
import os
from tqdm import tqdm

root_dir = "../artefacts/eval_audio/generate_tasks_2/"

# %%

# recursively find all subdirs
def find_midi_dirs(root_dir):
    midi_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if any(file.endswith(".mid") for file in files):
            midi_dirs.append(root)
    return midi_dirs




records = []
for dir in find_midi_dirs(root_dir):
    files = glob.glob(f"{dir}/*.mid", recursive=True)
    for file in tqdm(files):
        midi = muspy.read_midi(file)
        # check if midi is empty
        if len(midi.tracks)>0:
            record = {
                "set": dir,
                "pitch_range": muspy.pitch_range(midi),
                "n_pitches_used": muspy.n_pitches_used(midi),
                "n_pitch_classes_used": muspy.n_pitch_classes_used(midi),
                "polyphony": muspy.polyphony(midi),
                "polyphony_rate": muspy.polyphony_rate(midi),
                "scale_consistency": muspy.scale_consistency(midi),
                "pitch_entropy": muspy.pitch_entropy(midi),
                "pitch_class_entropy": muspy.pitch_class_entropy(midi),
                "empty_beat_rate": muspy.empty_beat_rate(midi),
                "drum_pattern_consistency": muspy.drum_pattern_consistency(midi),
            }
            records.append(record)   



# %%
        
import pandas as pd

df = pd.DataFrame(records)

# %%

df["set"] = df["set"].str.replace("../artefacts/eval_audio/generate_tasks_2/", "")

df = df.sort_values("set", ascending=False)

# remove ../artefacts/eval_audio/generate_tasks_2


# print dir averages and stds across metrics in a nice format
print(df.groupby("set").agg(["mean", "std"]).T)

# print a nice table
print(df.groupby("set").agg(["mean", "std"]).T.to_latex())

from IPython.display import display

display(df.groupby("set").agg(["mean", "std"]))

# put natural on top


#%%
# plot histograms of each metric 

import matplotlib.pyplot as plt

# for each metric, plot a histogram of the values for each set in a different color

for metric in df.columns[2:]:
    plt.figure()
    for set in df["set"].unique():
        plt.hist(df[df["set"] == set][metric], alpha=0.5, label=set)
    plt.title(metric)
    plt.legend()
    plt.show()
    




# %%
