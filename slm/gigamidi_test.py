#%%
from datasets import load_dataset
from tqdm import tqdm
import symusic

dataset = load_dataset("Metacreation/GigaMIDI", split="validation")

TPQ = 24

def preprocess_sample(sample):
    # remove the midi file from the sample
    score = symusic.Score.from_midi(sample["music"])
    # 
    # get resolution
    original_tpq = score.tpq
    score = score.resample(TPQ, min_dur=0)

    qpm = score.tempos[-1].qpm
    time_sig = f"{score.time_signatures[-1].numerator}/{score.time_signatures[-1].denominator}"


    loops = sample["loops"]
    loop_records = []
    for i in range(len(loops["track_idx"])):
        loop_record = {key: loops[key][i] for key in loops.keys()}
        loop_records.append(loop_record)

    # resample loop ticks to new tpq
    for loop in loop_records:
        loop["start_tick"] = loop["start_tick"] * TPQ // original_tpq
        loop["end_tick"] = loop["end_tick"] * TPQ // original_tpq

    return {**sample, "midi": score, "loops": loop_records, "qpm": qpm, "time_sig": time_sig}
    
records = []
for sample in tqdm(dataset):
    try:
        records.append(preprocess_sample(sample))
    except Exception as e:
        continue
print(f"Number of records: {len(records)}")

#%%

# look at top 10 most common qpm values
import pandas
qpm_counts = pandas.Series([record["qpm"] for record in records]).value_counts()
print(qpm_counts.head(10))

# look at top 10 most common time signatures
time_sig_counts = pandas.Series([record["time_sig"] for record in records]).value_counts()
print(time_sig_counts.head(20))




#%%

#%%
    
# get loop durations
# flatten into a list of loops
loop_list = []
for record in records:
    for loop in record["loops"]:
        loop["duration"] = loop["end_tick"] - loop["start_tick"]
        loop_list.append(loop)

#%%
# get histogram of loop durations
import matplotlib.pyplot as plt
durations = [loop["duration"] for loop in loop_list]
plt.hist(durations, bins=100)
plt.show()

#%%
# what are the loop durations?



#%%
print(records[0])




# %%
# look at how many loops we have
# print number of loops


# %%
