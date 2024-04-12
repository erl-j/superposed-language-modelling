#%%
import muspy
import glob
import os
from tqdm import tqdm
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

root_dir = "../artefacts/eval_cropped_midi/fad_test/"

#%%
import numpy as np

# embedding dir
embed_dir = "../artefacts/eval_audio/fad_test/"

embedding_name = "clap-laion-audio"

# read embeddings
embedding_paths = glob.glob(f"{embed_dir}/**/{embedding_name}/*.npy",recursive=True)

# read embeddings
records = []
for embedding_path in tqdm(embedding_paths):
    if "nr" in embedding_path:
        crop_path = embedding_path.replace(embed_dir,"")
        split_path = crop_path.split("/")
        task = split_path[0]

        embedding = np.load(embedding_path)

        if "natural" in task:
            record = {
                "path": embedding_path,
                "embedding": embedding,
                "task": task,
                "src_id": split_path[1].replace(".npy",""),
            }
        else:
            task,system,_d,_d,fn = split_path
            src_id = fn.replace(".npy","")
            record = {
                "path": embedding_path,
                "embedding": embedding,
                "task": task,
                "system": system,
                "src_id": src_id,
            }
        records.append(record)

#%%

src_embeds = []
for src_idx in range(100):
    src_embed = np.load(f"../artefacts/eval_audio/fad_test/natural/embeddings/clap-laion-audio/nr_{src_idx}.npy")
    src_embeds.append(np.mean(src_embed, axis=0))
    
src_embeds = np.stack(src_embeds)



editing_tasks = [
    "infilling_start",
    "infilling_end",
    "infilling_low",
    "pitch_set",
    "constrained_generation",
    "generate",
    "infilling_high_patched",
    "infilling_box_middle",
]

temperature = "1.0"
for task in editing_tasks:
    print(task)
    # load slm embeddings
    slm_embeds = []
    for src_idx in range(100):
        path = f"../artefacts/eval_audio/fad_test/{task}/*slm_t={temperature}/embeddings/clap-laion-audio/nr_{src_idx}.npy"
        gen_embed = np.load(glob.glob(path)[0])
        slm_embeds.append(np.mean(gen_embed, axis=0))
    slm_embeds = np.stack(slm_embeds)

    # l2 normalize
    norm_slm_embeds = slm_embeds / np.linalg.norm(slm_embeds, axis=1)[:,None]

    mlm_embeds = []
    for src_idx in range(100):
        path = f"../artefacts/eval_audio/fad_test/{task}/*mlm_t={temperature}/embeddings/clap-laion-audio/nr_{src_idx}.npy"
        gen_embed = np.load(glob.glob(path)[0])
        mlm_embeds.append(np.mean(gen_embed, axis=0))

    mlm_embeds = np.stack(mlm_embeds)
    # l2 norm
    norm_mlm_embeds = mlm_embeds / np.linalg.norm(mlm_embeds, axis=1)[:,None]

    # l2 normalize
    norm_src_embeds = src_embeds / np.linalg.norm(src_embeds, axis=1)[:,None]

    # get cosine similarity matrix
    slm_cosine_sim = np.dot(norm_slm_embeds, norm_src_embeds.T)
    slm_cosine_sim_diag = np.diag(slm_cosine_sim)
    mlm_cosine_sim = np.dot(norm_mlm_embeds, norm_src_embeds.T)
    mlm_cosine_sim_diag = np.diag(mlm_cosine_sim)
    nat_self_cosine_sim = np.dot(norm_src_embeds, norm_src_embeds.T)
    nat_self_cosine_sim_diag = np.diag(nat_self_cosine_sim)

    # plot histograms of cosine similarity
    plt.figure()
    n_bins = 10
    plt.hist(slm_cosine_sim_diag, bins=n_bins, alpha=0.5, label="slm")
    plt.hist(mlm_cosine_sim_diag, bins=n_bins, alpha=0.5, label="mlm")
    plt.hist(nat_self_cosine_sim_diag, bins=n_bins, alpha=0.5, label="natural")
    # plot means
    plt.axvline(np.mean(slm_cosine_sim_diag), color="blue", linestyle="dashed")
    plt.axvline(np.mean(mlm_cosine_sim_diag), color="orange", linestyle="dashed")
    plt.axvline(np.mean(nat_self_cosine_sim_diag), color="green", linestyle="dashed")
    plt.title(f"{task} cosine similarity")
    plt.legend()
    plt.show()

    # count slm wins
    slm_wins = np.sum(slm_cosine_sim_diag > mlm_cosine_sim_diag)
    print(f"slm wins: {slm_wins}/{len(slm_cosine_sim_diag)}")


    # plot both histograms with different alphas
    





#%%


for task in editing_tasks:
    for system in ["mlm", "slm"]:
        # get embedding for task
        for src_id in range(100):
            # get natural embedding for src_id
            src_embed = df
   

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



#%%
df = pd.DataFrame(records)
df["set"] = df["set"].str.replace("../artefacts/eval_cropped_midi/fad_test/", "")
# split set into task and system
df["task"] = df["set"].str.split("/").str[0]
# drop tasks infilling_drums and infilling_harmonic
df = df[~df["task"].str.contains("infilling_drums")]
df = df[~df["task"].str.contains("infilling_harmonic")]
df["system_tmp"] = df["set"].str.split("/").str[1]
df = df.drop(columns=["set"])

# split system_tmp into system and temperature
df["system"] = df["system_tmp"].str.split("t=").str[0]
df["temperature"] = df["system_tmp"].str.split("t=").str[-1]
# replace "_slm_" with "slm" and "_mlm_" with "mlm"
df["system"].replace("_slm_", "slm", inplace=True)
df["system"].replace("_mlm_", "mlm", inplace=True)
df = df.drop(columns=["system_tmp"])


df

# print unique systems
display(df)

temperature = "1.0"
for task in df["task"].unique():
    for metric in df.columns:
        if task in ["natural", "natural2"]:
            continue
        if metric == "task" or metric == "system" or metric == "temperature":
            continue
        # first plot a histogram
        # for each system plot a histogram of the metric
        # alo plot for task natural
        # set n_bins to 10
        plt.figure()
        n_bins = 10
        plt.hist(df[(df["task"] == "natural2")] [metric], label="natural2", alpha=0.5)
        plt.hist(df[(df["task"] == "natural")] [metric], label="natural", alpha=0.5, color="red")
        for system in ["mlm", "slm"]:
            plt.hist(df[(df["task"] == task) & (df["system"] == system) & (df["temperature"] == temperature)][metric], label=system, alpha=0.5)
        plt.title(f"{task} {metric}")
        plt.legend()

#%%
df = df.sort_values("set", ascending=False)
# remove ../artefacts/eval_audio/generate_tasks_2
pd.options.display.float_format = '{:.2f}'.format
df = df.groupby("task","system").agg(["mean", "std"])
# highlight means with least distance to natural in red
df
# plot histograms of each metric 




#%%
# read fad results
def parse_fad_line(line):
    parts = line.split()
    reference = parts[5]
    system = parts[7]
    score = float(parts[-1])
    ref = reference.replace("artefacts/eval_audio/fad_test/","")
    return {"system": system.replace("artefacts/eval_audio/fad_test/",""), parts[2]+"_fad_"+ref: score }

fad_a = open("../fad_results_clap_audio_new.txt").readlines()
fad_a = [x.strip() for x in fad_a]
fad_a = [parse_fad_line(x) for x in fad_a]
fad_a = sorted(fad_a, key=lambda x: x["system"])
fad_a = [dict(x, **y) for x, y in zip(fad_a[::2], fad_a[1::2])]
fad_df = pd.DataFrame(fad_a)
fad_df





# remove rows where "natural" is in the system name, avoiding bad operand type for unary ~: 'str'
fad_df = fad_df[~fad_df["system"].str.contains("natural")]
# remove rows with "infilling_drums" and "infilling_harmonic" in the system name
fad_df = fad_df[~fad_df["system"].str.contains("infilling_drums")]
fad_df = fad_df[~fad_df["system"].str.contains("infilling_harmonic")]

# decompose system into task and system
fad_df["task"] = fad_df["system"].str.split("/").str[0]
fad_df["system"] = fad_df["system"].str.split("/").str[1]

display(fad_df)



# split system into system and temperature
fad_df["temperature"] = fad_df["system"].str.split("t=").str[-1]

# convert temperature to float
fad_df["temperature"] = fad_df["temperature"].astype(float)
fad_df["system"] = fad_df["system"].str.split("t=").str[0]


# show lines with highest scores

# for each task, plot a line plot with the scores for each system with temperature on x
# scores are either clap-laion-music_fad_natural or clap-laion-audio_fad_natural2
# make 
for task in fad_df["task"].unique():
    plt.figure()
    for system in fad_df[fad_df["task"] == task]["system"].unique():
        plt.plot(fad_df[(fad_df["task"] == task) & (fad_df["system"] == system)]["temperature"], fad_df[(fad_df["task"] == task) & (fad_df["system"] == system)]["clap-laion-audio_fad_natural"], label=system, linestyle="dashed", color="red" if "mlm" in system else "green")
        plt.plot(fad_df[(fad_df["task"] == task) & (fad_df["system"] == system)]["temperature"], fad_df[(fad_df["task"] == task) & (fad_df["system"] == system)]["clap-laion-audio_fad_natural2"], label=system, linestyle="solid", color="red" if "mlm" in system else "green")

    plt.title(task)
    plt.legend()
    plt.show()
#%%
fad_b = [parse_fad_line(x) for x in fad_b]
fad_b = sorted(fad_b, key=lambda x: x["system"])
fad_b = [dict(x, **y) for x, y in zip(fad_b[::2], fad_b[1::2])]
fad_df_b = pd.DataFrame(fad_b)

# merge on system
fad_df = pd.merge(fad_df, fad_df_b, on="system")
fad_df
# %%
        



# %%
