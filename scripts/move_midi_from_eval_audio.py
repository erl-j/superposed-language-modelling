import glob
import os
root = "artefacts/eval_audio"

# find all midi files
midi_files = glob.glob(f"{root}/**/*.mid", recursive=True)

midi_root = "artefacts/eval_cropped_midi"

# move all midi files to the new root, preserving the folder structure
for midi_file in midi_files:
    new_path = os.path.join(midi_root, os.path.relpath(midi_file, root))
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    os.rename(midi_file, new_path)
    print(f"Moved {midi_file} to {new_path}")



