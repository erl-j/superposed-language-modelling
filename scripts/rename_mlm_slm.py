# we accidentally mislabeled the models for the first run
# this scrips renames the outputs of the models

import os
# first rename all directiories with mlm in the name to instead have _slm in the name
ROOT_DIR = "./artefacts/eval/generate_tasks_2/"

for root, dirs, files in os.walk(ROOT_DIR):
    for d in dirs:
        if d.startswith("slm"):
            new_d = d.replace("slm", "_mlm")
            os.rename(os.path.join(root, d), os.path.join(root, new_d))
            print(f"Renamed {d} to {new_d}")

