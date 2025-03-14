
#%%
import symusic
import glob
# import rich progress bar
import rich
import os
from tqdm import tqdm
import pandas as pd

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
    fmd_df.to_csv(f"{path}/fmd.csv", index=False)


#%%
# save the fmd records to a dataframe


#%%
# load csv
fmd_df = pd.read_csv(f"{path}/fmd.csv")

# Optional: filter systems if you want to uncomment this
fmd_df = fmd_df[fmd_df["system_name"].str.contains("mlm|slm_mixed")]

# Identify ground truth examples - assuming they're marked by "ground_truth" in system_name
ground_truth_df = df[df["system_name"] == "ground_truth"]

# Count examples per task
# how many ground truth examples exist for each task
task_counts = ground_truth_df["task_name"].value_counts().to_dict()

print(task_counts)


# Get task names sorted by their count in ground truth
sorted_tasks = sorted(task_counts.keys(), key=lambda x: task_counts[x], reverse=True)

## sort the fmd_df by the sorted tasks
fmd_df = fmd_df.set_index("task_name").loc[sorted_tasks].reset_index()



# For debugging: print tasks and their counts
for task in sorted_tasks:
    print(f"Task: {task}, Count: {task_counts[task]}")

# Use these sorted tasks for the plot
n_systems = len(fmd_df["system_name"].unique())
n_tasks = len(fmd_df["task_name"].unique())

# show line plot of fmd for each system across tasks, with tasks sorted by example count
plt.figure(figsize=(20, 5))

# Use sorted_tasks for the order parameter to control task ordering on x-axis
sns.lineplot(
    data=fmd_df, 
    x="task_name", 
    y="fmd", 
    hue="system_name", 
)

plt.title("FMD by System and Task (Tasks Sorted by Ground Truth Example Count)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%

# add task type to fmd_df


#%% print scatter plot with x being fmd, y being how many ground truth examples exist for that task
# colour denotes system
import matplotlib.pyplot as plt
import seaborn as sns

# add a column to fmd_df with the number of ground truth examples for each task
task_to_count = {task: count for task, count in zip(sorted_tasks, [task_counts[task] for task in sorted_tasks])}
fmd_df["n_ground_truth"] = fmd_df["task_name"].apply(lambda x: task_to_count[x])

plt.figure(figsize=(10, 10))
sns.scatterplot(
    data=fmd_df, 
    y="fmd", 
    x="n_ground_truth", 
    hue="system_name"
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
