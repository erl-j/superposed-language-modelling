#%%
import symusic
import glob
from tqdm import tqdm

a_path = "../artefacts/applications_250/mlm_150epochs_left_to_right_reverse/strings_and_flute"
b_path = "../artefacts/applications_250/slm_mixed_150epochs_left_to_right_reverse/strings_and_flute"
c_path = "../artefacts/applications_250/ground_truth/"

model_dirs = [a_path, b_path, c_path]

# get all midi files
records = []
for model_dir in model_dirs:
    for midi_file in glob.glob(f"{model_dir}/**/*.mid", recursive=True):
        records.append({
            "task": model_dir.split("/")[-1],
            "model": model_dir.split("/")[-2] if model_dir.split("/")[-1] != "ground_truth" else "ground_truth",
            "midi_path": midi_file
        })

# load midi with symusic
records = [ {**record, "midi": symusic.Score(record["midi_path"])} for record in tqdm(records)]

print(len(records))
#%%
# now get list of pitches
def get_pitches(sm, drums):
    sm = sm.copy()
    pitches = []
    for track in sm.tracks:
        if track.is_drum and drums == "exclude":
            continue
        if drums == "only" and not track.is_drum:
            continue
        for note in track.notes:
            pitches.append(note.pitch)
    return pitches

DRUMS = "exclude"

records = [ {**record, "pitches": get_pitches(record["midi"], DRUMS)} for record in tqdm(records)]

# get total list of pitches for each model
a_pitches = [pitch for record in records for pitch in record["pitches"] if "mlm" in record["model"]]
b_pitches = [pitch for record in records for pitch in record["pitches"] if "slm" in record["model"]]
c_pitches = [pitch for record in records for pitch in record["pitches"] if "ground_truth" in record["model"]]   
# now create a histogram of pitches with labels
import numpy as np
import matplotlib.pyplot as plt

a_hist = np.histogram(a_pitches, bins=range(128))
b_hist = np.histogram(b_pitches, bins=range(128))
c_hist = np.histogram(c_pitches, bins=range(128))

plt.bar(a_hist[1][:-1], a_hist[0], alpha=0.5, label="mlm")
plt.bar(b_hist[1][:-1], b_hist[0], alpha=0.5, label="slm")
plt.bar(c_hist[1][:-1], c_hist[0], alpha=0.5, label="ground_truth")
plt.legend()



    


# %%
