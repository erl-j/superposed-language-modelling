import os
from util import render_directory_with_fluidsynth

root_dir = "artefacts/eval/generate_tasks_harmonic"

# recursively find all directories which contain midi files at next level
def find_midi_dirs(root_dir):
    midi_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if any(file.endswith(".mid") for file in files):
            midi_dirs.append(root)
    return midi_dirs

midi_dirs = find_midi_dirs(root_dir)

# render all directories with fluidsynth
for midi_dir in midi_dirs:
    render_directory_with_fluidsynth(midi_dir, midi_dir + "_audio")












