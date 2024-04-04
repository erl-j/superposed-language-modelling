#%%
import muspy
import glob

dir_a = "../artefacts/object_eval/natural"

dir_b = "../artefacts/object_eval/mlm_temp_1.0"

records = []
for dir in [dir_a, dir_b]:
    files = glob.glob(f"{dir}/**/*.mid", recursive=True)
    for file in files:
        midi = muspy.read_midi(file)
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
        }
        records.append(record)   



# %%
        
import pandas as pd

df = pd.DataFrame(records)

# %%

# print dir averages and stds across metrics in a nice format
print(df.groupby("set").agg(["mean", "std"]).T)


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
