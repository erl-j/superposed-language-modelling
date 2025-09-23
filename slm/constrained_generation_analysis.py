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
path = "../artefacts/constrained_generation_5"
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
    fmd_df.to_csv(f"{path}/fmd5.csv", index=False)

#%%
# load csv
fmd_df = pd.read_csv(f"{path}/fmd5.csv")

# system name is like this:
# slm_mixed_150epochs_order=left_to_right_reverse_t=1.0_set_n_notes=False
# <model>_<n_epochs>_order=<order>_t=<t>_set_n_notes=<set_n_notes>
# either "mlm_", "slm_mixed_" or "slm_full"
# everything before "150epochs"
# print unique system names
print(fmd_df["system_name"].unique())
fmd_df["model"] = fmd_df["system_name"].apply(lambda x: x.split("_150epochs")[0])
fmd_df["order"] = fmd_df["system_name"].str.extract(r'order=(.*?)_t=')
fmd_df["t"] = fmd_df["system_name"].str.extract(r't=(.*?)_set_n_notes=')
fmd_df["set_n_notes"] = fmd_df["system_name"].str.extract(r'set_n_notes=(.*)')#%%

#%%

# make a bar plot of fmd by model and task
# sort by task, order, model

task_name_to_display_name = {
    "c major pitch set": "a) pitch: C major",
    "c pentatonic": "b) pitch: C pentatonic",
    "1 d 16 notes": "c) onset: 16th notes",
    "1 d 8 notes": "d) onset: 8th notes",
    "1 d 4 notes": "e) onset: quarter notes",
    # "1 d 2 notes": "f) onset: half notes",
    "guitar": "f) instr: guitar",
    "piano": "g) instr: piano",
    "drum_and_bass": "h) instr: drum and bass",
    "drum_and_bass_and_guitar": "i) instr: drums, bass, guitar",
    "drum_and_bass_and_piano": "j) instr: drums, bass, piano, guitar",
    "drums": "k) instr: drums",
}

# remove half notes
fmd_df = fmd_df[fmd_df["task_name"] != "1 d 2 notes"]

# model order is mlm, full, mixed sparse
model_order = ["mlm", "slm_full", "slm_mixed", "slm_sparse"]

fmd_df["display_task_name"] = fmd_df["task_name"].apply(lambda x: task_name_to_display_name[x])

fmd_df = fmd_df.sort_values(by=["task_name"])

# sort by model
fmd_df = fmd_df.sort_values(by="model", key=lambda x: pd.Categorical(x, categories=model_order, ordered=True))
fmd_df = fmd_df.sort_values(by="display_task_name")

for set_n_notes in [True, False]:

    # first for set_n_notes=True
    fmd_df_set_n_notes = fmd_df[fmd_df["set_n_notes"] == str(set_n_notes)]

    # sort in task name to display order
    # let me pick colours for the models
    model_colours = {
        "mlm": "#E63946",      # coral red
        "slm_full": "#FFB703", # warm yellow
        "slm_mixed": "#2A9D8F", # teal green  
        "slm_sparse": "#457B9D" # steel blue
    }
    # Plot for random sampling
    data_random = fmd_df_set_n_notes[fmd_df_set_n_notes["order"] == "random"]
    plt.figure(figsize=(20, 10))

    ax = sns.barplot(
        data=data_random,
        x="display_task_name",
        y="fmd",
        hue="model", 
        hue_order=model_order,
        palette=model_colours,
        alpha=1.0
    )

    # Add left-to-right and right-to-left values above bars
    data_ltr = fmd_df_set_n_notes[fmd_df_set_n_notes["order"] == "left_to_right"]
    data_rtl = fmd_df_set_n_notes[fmd_df_set_n_notes["order"] == "left_to_right_reverse"]

    # Get bar positions and widths from the plot
    bars = ax.patches
    n_tasks = len(data_random["display_task_name"].unique())
    n_models = len(model_order)
    width = bars[0].get_width()

    # Add arrows above bars
    for i in range(n_tasks):
        for j, model in enumerate(model_order):
            # Get the bar index
            idx = i * n_models + j
            
            # Get the bar center x-coordinate
            x = bars[idx].get_x() + width/2
            
            # Get FMD values for different orders
            ltr_vals = data_ltr[(data_ltr["display_task_name"]==data_random["display_task_name"].unique()[i]) & 
                            (data_ltr["model"]==model)]["fmd"]
            rtl_vals = data_rtl[(data_rtl["display_task_name"]==data_random["display_task_name"].unique()[i]) & 
                            (data_rtl["model"]==model)]["fmd"]
            
            # Draw arrows at their actual FMD values if values exist
            if len(ltr_vals) > 0:
                plt.text(x-0.01, ltr_vals.iloc[0], "←", ha='center', va='bottom', fontsize=16)
            if len(rtl_vals) > 0:
                plt.text(x+0.01, rtl_vals.iloc[0], "→", ha='center', va='bottom', fontsize=16)

    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 500)  # Set y-axis limits from 0 to 300
    plt.title(f"Fixed number of active events: {set_n_notes}")
    plt.xlabel("")  # Remove x-axis label
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    # save the plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/fmd_plot_{set_n_notes}.png", dpi=300)
    plt.show()



#%% now a plot to compare the three sampling orders.
plt.figure(figsize=(30, 10))

# Create markers for each sampling order
markers = {'left_to_right': '$→$', 'left_to_right_reverse': '$←$', 'random': '$⚄$'}

data = fmd_df_set_n_notes
unique_tasks = data['display_task_name'].unique()
task_indices = {task: idx for idx, task in enumerate(unique_tasks)}

# Create a strip plot with custom markers
for order_idx, order in enumerate(markers.keys()):
    order_data = data[data['order'] == order]
    for model_idx, model in enumerate(model_order):
        model_data = order_data[order_data['model'] == model]
        # Add horizontal offset based on order index and model index
        x_positions = [task_indices[task] + (order_idx - 1) * 0.2 + model_idx * 0.5 for task in model_data['display_task_name']]
        plt.scatter(
            x_positions,
            model_data['fmd'],
            marker=markers[order],
            c=[model_colours[model]], 
            s=200,
            label=f"{model} ({order})"
        )
plt.xticks(range(len(unique_tasks)), unique_tasks, rotation=45, ha='right')
plt.ylim(0, 300)  # Set y-axis limits from 0 to 500
plt.title("Sampling Order Comparison (Fixed number of active events)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%% add sampling order to the plot
# arrow left to right
# arrow right to left
# and dice



# now we do variable number of active events
fmd_df_not_set_n_notes = fmd_df[fmd_df["set_n_notes"] == "False"]

# same plot as above
plt.figure(figsize=(20, 5))
data = fmd_df_not_set_n_notes[fmd_df_not_set_n_notes["order"] == "random"]
sns.barplot(
    data=data, 
    x="display_task_name", 
    y="fmd", 
    hue="model",
    hue_order=model_order,
    palette=model_colours,
)
plt.xticks(rotation=45, ha='right')
plt.title(f"Variable number of active events, random sampling order")
plt.show()

#%%
#%%

# make a barplot per sampling order
for order in fmd_df_set_n_notes["order"].unique():
    plt.figure(figsize=(20, 5))
    data = fmd_df_set_n_notes[fmd_df_set_n_notes["order"] == order]
    sns.barplot(
        data=data, 
        x="display_task_name", 
        y="fmd", 
        hue="model",
        hue_order=model_order,
        palette=model_colours,
    )
    plt.title(f"Sampling order: {order}, Fixed number of active events")
    plt.show()

# now for set_n_notes=False
fmd_df_not_set_n_notes = fmd_df[fmd_df["set_n_notes"] == "False"]

# make a barplot per sampling order


for order in fmd_df_not_set_n_notes["order"].unique():
    plt.figure(figsize=(20, 5))
    data = fmd_df_not_set_n_notes[fmd_df_not_set_n_notes["order"] == order]
    sns.barplot(
        data=data, 
        x="display_task_name", 
        y="fmd", 
        hue="model",
    )
    plt.title(f"Sampling order: {order}, Variable number of active events")
    plt.show()

#%%
# then for set_n_notes=False
fmd_df_not_set_n_notes = fmd_df[fmd_df["set_n_notes"] == "False"]



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
# plt.title("FMD w.r.t. to ground truth, no fixed number of active events")
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
