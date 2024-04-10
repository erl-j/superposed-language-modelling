import datetime
import os

# zip src
zip_src = "artefacts/eval_cropped_midi/fad_test"
# Check if the directory exists and contains files
if os.path.exists(zip_src) and os.listdir(zip_src):
    # zip dest
    zip_dest = "artefacts/demo_midi.zip"
    # zip command
    zip_command = f"zip -r {zip_dest} {zip_src}"
    # run command
    os.system(zip_command)
else:
    print("No files found in the source directory.")
