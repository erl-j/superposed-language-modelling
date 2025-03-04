import os
import symusic
import util
from tqdm import tqdm

root = "artefacts/applications_250e"

loops = 4

# copy directory
os.system(f"cp -r {root} {root}_{loops}x")

dest_root = f"{root}_{loops}x"

import glob

# find all .mid files
mid_files = glob.glob(f"{dest_root}/**/*.mid", recursive=True)

# loop over all .mid files
for mid_file in tqdm(mid_files):

    midi = symusic.Score(mid_file)
    
    loop_midi = util.loop_sm(midi, 4, loops)

    loop_midi.dump_midi(mid_file)
    


