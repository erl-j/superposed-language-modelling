#%%
from musicaiz import loaders
import glob
from tqdm import tqdm
midi_path = "../artefacts/constrained_generation_2"
midi_paths = glob.glob(f"{midi_path}/**/*.mid", recursive=True)

def try_except(func, default):
    try:
        return func()
    except Exception:
        return default

records = [{"path": path, "system": path.split("/")[-3], "task": path.split("/")[-2]} for path in tqdm(midi_paths)]

for i in tqdm(range(len(records))):
    try:
        records[i]["musa"] = loaders.Musa(records[i]["path"])
    except Exception:
        records[i]["musa"] = None

#%%

# 



           
#%%
# Load the data

