import os
import shutil
import glob
import json
from pathlib import Path

# Define directories
midi_dir = Path("../artefacts/applications")
public_dir = Path("public")

# Create public directory if it doesn't exist
public_dir.mkdir(exist_ok=True)

# Remove existing content in public_dir if it exists
if public_dir.exists():
    shutil.rmtree(public_dir)

# Copy all files from midi_dir to public_dir
shutil.copytree(midi_dir, public_dir)

midi_files = glob.glob(str(midi_dir / "**/*.mid"), recursive=True)

# every file that has ground_truth in its name is a ground truth file
# ground_truth/example_{example_number}.mid

# otherwise, it's structured as follows:
# {model_name}/{task_name}/generated_{example_number}.mid

# create records for each file with model, task, and example number.
# ground truth files have model ground truth and task as None

records = []

for file in midi_files:
    if "ground_truth" in file:
        model = "ground_truth"
        task = None
        example_number = int(file.split("_")[-1].split(".")[0])
    else:
        model, task, example_number = file.split("/")[-3:]
        example_number = int(example_number.split("_")[-1].split(".")[0])
    
    records.append({
        "model": model,
        "task": task,
        "example_number": example_number,
        "file": file.replace("../artefacts/applications/", "")
    })

# Save records to public directory
with open(public_dir / "records.json", "w") as f:
    json.dump(records, f, indent=4)



