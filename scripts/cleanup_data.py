#%%
from data import MidiDataset
ROOT_DIR = "../"
TMP_DIR = ROOT_DIR + "artefacts/tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/output"

dataset = "mmd_loops"

ROOT_DIR = "../"
TMP_DIR = ROOT_DIR + "artefacts/tmp"
OUTPUT_DIR = ROOT_DIR + "artefacts/output"

MODEL_BARS = 4
# Load the dataset
ds = MidiDataset(
    cache_path=ROOT_DIR+f"data/{dataset}/tst_midi_records_unique_pr.pt",
    path_filter_fn=lambda x: f"n_bars={MODEL_BARS}" in x if dataset=="mmd_loops" else True,
    genre_list=ROOT_DIR+"data/mmd_loops/genres.txt",
)
#%%
print(len(ds.records))
print(ds.records[0])

#%%

records = ds.records
import muspy as mp
from tqdm import tqdm

new_records = []
for i in tqdm(range(len(ds.records))):
    m = mp.read_midi(ds.records[i]["path"].replace("../","../../"))
    tpb = m.resolution
    tpm = tpb * 4
    metrics = {
        "pitch_range":mp.pitch_range(m),
        "n_pitches_used":mp.n_pitches_used(m),
        "n_pitch_classes_used":mp.n_pitch_classes_used(m),
        "polyphony":mp.polyphony(m),
        "polyphony_rate":mp.polyphony_rate(m),
        "scale_consistency":mp.scale_consistency(m),
        "pitch_entropy":mp.pitch_entropy(m),
        "pitch_class_entropy":mp.pitch_class_entropy(m),
        "empty_beat_rate":mp.empty_beat_rate(m),
        "drum_in_pattern_rate":mp.drum_in_pattern_rate(m,"duple"),
        "drum_pattern_consistency":mp.drum_pattern_consistency(m),
        "groove_consistency":mp.groove_consistency(m,tpm),
        "empty_measure_rate":mp.empty_measure_rate(m,tpm),
    }
    new_records.append({**ds.records[i],**metrics})
# %%

# plot histograms for all metrics
import seaborn as sns
import matplotlib.pyplot as plt

metrics=["pitch_range","n_pitches_used","n_pitch_classes_used","polyphony","polyphony_rate","scale_consistency","pitch_entropy","pitch_class_entropy","empty_beat_rate","drum_in_pattern_rate","drum_pattern_consistency","groove_consistency","empty_measure_rate"]
for metric in metrics:
    sns.histplot([x[metric] for x in new_records])
    plt.title(metric)
    plt.show()



# %%

# %%
