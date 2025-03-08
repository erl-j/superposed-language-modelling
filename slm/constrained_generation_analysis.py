
#%%
import symusic
import glob
# import rich progress bar
import rich
import os
from tqdm import tqdm
import pandas as pd

#%%
path = "../artefacts/constrained_generation"
midi_paths = glob.glob(f"{path}/**/*.mid", recursive=True)
midi_paths = sorted(midi_paths)

records = [{"path": path, "score": symusic.Score(path)} for path in tqdm(midi_paths)]

df = pd.DataFrame(records)
# %%
# print first path
# split the path by '/'. <system_name>/<task_name>/<midi_file_name>

df["system_name"] = df["path"].apply(lambda x: x.split("/")[-3])
df["task_name"] = df["path"].apply(lambda x: x.split("/")[-2])

df["n_notes"] = df["score"].apply(lambda x: x.note_num())

#%%
# for each system and task show the note number histogram in subplots
import matplotlib.pyplot as plt
import seaborn as sns

n_systems = len(df["system_name"].unique())
n_tasks = len(df["task_name"].unique())
fig, axes = plt.subplots(n_systems, n_tasks, figsize=(20, 20))

#%% 

max_notes = 300
min_notes = 0

# for each system, show the note number histogram across tasks,
for i, system in enumerate(df["system_name"].unique()):
    fig, axes = plt.subplots(1, n_tasks, figsize=(20, 5))
    for j, task in enumerate(df["task_name"].unique()):
        data = df[(df["system_name"] == system) & (df["task_name"] == task)]
        sns.histplot(data["n_notes"], ax=axes[j])
        axes[j].set_title(f"{task}")
        axes[j].set_xlim(min_notes, max_notes)
    fig.suptitle(f"{system}")
    plt.show()

#%%

# for each task, show some midi examples from each system
from util import preview_sm

n_systems = len(df["system_name"].unique())
n_tasks = len(df["task_name"].unique())

task = "c pentatonic"
print(f"Task: {task}")
for system in df["system_name"].unique():
    print(f"System: {system}")
    data = df[(df["system_name"] == system) & (df["task_name"] == task)]
    for i in range(1):
        preview_sm(data.iloc[i]["score"])


# %%
