
#%%
import symusic
import glob
# import rich progress bar
import rich
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
path = "../artefacts/constrained_generation_2"
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
COMPUTE_FMD = False
if COMPUTE_FMD:
    from frechet_music_distance import FrechetMusicDistance
    from frechet_music_distance.utils import clear_cache
    feature_extractor = 'clamp2'  # Could make this a parameter
    clear_cache()
    metric = FrechetMusicDistance(
        feature_extractor=feature_extractor, 
        gaussian_estimator='mle', 
        verbose=True
    )

    fmd_records = []

    for system in df["system_name"].unique():
        for task in df["task_name"].unique():
            print(f"System: {system}, Task: {task}")

            # check if test path has .mid files
            # Check if test path has .mid files
            test_path = f"{path}/{system}/{task}"
            if not os.path.exists(test_path):
                print(f"Skipping {test_path} - path does not exist")
                continue
                
            test_midi_files = glob.glob(f"{test_path}/*.mid")
            if len(test_midi_files) == 0:
                print(f"Skipping {test_path} - no MIDI files found")
                continue
            
            score = metric.score(
                reference_path=f"{path}/ground_truth/{task}",
                test_path=f"{path}/{system}/{task}"
            )
            fmd_records.append({
                "system_name": system,
                "task_name": task,
                "fmd": score
            })
            print(f"{system}/{task}: {score}")

    fmd_df = pd.DataFrame(fmd_records)
    # save the dataframe
    fmd_df.to_csv(f"{path}/fmd3.csv", index=False)

#%%
# load csv
fmd_df = pd.read_csv(f"{path}/fmd3.csv")

# print columns
print(fmd_df.columns)
#%%

# Optional: filter systems if you want to uncomment this


# # Identify ground truth examples - assuming they're marked by "ground_truth" in system_name
# ground_truth_df = df[df["system_name"] == "ground_truth"]

# # Count examples per task
# # how many ground truth examples exist for each task
# task_counts = ground_truth_df["task_name"].value_counts().to_dict()

# print(task_counts)


# # Get task names sorted by their count in ground truth
# sorted_tasks = sorted(task_counts.keys(), key=lambda x: task_counts[x], reverse=True)

## sort the fmd_df by the sorted tasks
# fmd_df = fmd_df.set_index("task_name").loc[sorted_tasks].reset_index()



# # For debugging: print tasks and their counts
# for task in sorted_tasks:
#     print(f"Task: {task}, Count: {task_counts[task]}")

# Use these sorted tasks for the plot
n_systems = len(fmd_df["system_name"].unique())
n_tasks = len(fmd_df["task_name"].unique())

# show line plot of fmd for each system across tasks, with tasks sorted by example count
plt.figure(figsize=(20, 5))

# # Use sorted_tasks for the order parameter to control task ordering on x-axis
# sns.lineplot(
#     data=fmd_df, 
#     x="task_name", 
#     y="fmd", 
#     hue="system_name", 
# )

# plt.title("FMD by System and Task (Tasks Sorted by Ground Truth Example Count)")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()



task_name_to_display_name = {
    "c major pitch set": "pitch: C major",
    "c pentatonic": "pitch: C pentatonic",
    "1 d 16 notes": "onset: 16th notes",
    "1 d 8 notes": "onset: 8th notes",
    "1 d 4 notes": "onset: quarter notes",
    "1 d 2 notes": "onset: half notes",
    "guitar": "instr: guitar",
    "piano": "instr: piano",
    "drum_and_bass": "instr: drum and bass",
    "drum_and_bass_and_guitar": "instr: drums, bass, guitar",
    "drum_and_bass_and_piano": "instr: drum and bass and piano and guitar",
    "drums": "instr: drums",
}

fmd_df["display_task_name"] = fmd_df["task_name"].apply(lambda x: task_name_to_display_name[x])
fmd_df["task_type"] = fmd_df["display_task_name"].apply(lambda x: x.split(":")[0])







# show bar plot of fmd for each system across tasks, with tasks sorted by task type and example count
plt.figure(figsize=(20, 5))

fmd_df = fmd_df.sort_values(by=["system_name","task_type" ], ascending=[True, False])

# only consider systems that have mlm or mixed in their name
fmd_df = fmd_df[fmd_df["system_name"].str.contains("mlm|mixed")]
# filter away systems that have =False in them
fmd_df = fmd_df[~fmd_df["system_name"].str.contains("=False")]
# print unique system names
print("Unique system names:")
print(fmd_df["system_name"].unique())


# filter away rows where system_name contains "set_n_notes"
fmd_df_set_n_notes = fmd_df[~fmd_df["system_name"].str.contains("set_n_notes")]




sns.barplot(
    data=fmd_df_set_n_notes, 
    x="display_task_name", 
    y="fmd", 
    hue="system_name", 
)
plt.title("FMD w.r.t. to ground truth, fixed number of active events")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# now filter away rows where system_name does not contain "set_n_notes"
fmd_df_not_set_n_notes = fmd_df[fmd_df["system_name"].str.contains("set_n_notes")]

plt.figure(figsize=(20, 5))
sns.barplot(
    data=fmd_df_not_set_n_notes, 
    x="display_task_name", 
    y="fmd", 
    hue="system_name", 
)
plt.title("FMD w.r.t. to ground truth, no fixed number of active events")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%%

# print unique tasks
print(fmd_df["task_name"].unique())

#%%

task_types = fmd_df["task_type"].unique()
n_task_types = len(task_types)

plt.figure(figsize=(20, 5))

for i, task_type in enumerate(task_types):
    plt.subplot(1, n_task_types, i+1)
    data = fmd_df[fmd_df["task_type"] == task_type]
    sns.lineplot(
        data=data, 
        x="display_task_name", 
        y="fmd", 
        hue="system_name", 
    )
    plt.title(task_type)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% print scatter plot with x being fmd, y being how many ground truth examples exist for that task
# colour denotes system
import matplotlib.pyplot as plt
import seaborn as sns

# add a column to fmd_df with the number of ground truth examples for each task
task_to_count = {task: count for task, count in zip(sorted_tasks, [task_counts[task] for task in sorted_tasks])}
fmd_df["n_ground_truth"] = fmd_df["task_name"].apply(lambda x: task_to_count[x])

import math
fmd_df["log_n_ground_truth"] = fmd_df["n_ground_truth"].apply(lambda x: math.log(x))
# log scale for x-axis
plt.figure(figsize=(10, 10))
sns.scatterplot(
    data=fmd_df, 
    y="fmd", 
    x="log_n_ground_truth", 
    hue="system_name",

)
plt.title("FMD vs Number of Ground Truth Examples")
plt.show()



#%% 




#%%
# for each example show how many ground truth examples exist



# count number of examples per task




#%%



            


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

task = "c major pitch set"
print(f"Task: {task}")
for system in df["system_name"].unique():
    print(f"System: {system}")
    data = df[(df["system_name"] == system) & (df["task_name"] == task)]
    for i in range(1):
        preview_sm(data.iloc[i]["score"])


# %%
