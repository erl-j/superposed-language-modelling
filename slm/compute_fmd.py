from frechet_music_distance import FrechetMusicDistance
import os

def find_midi_dirs(base_path):
   midi_dirs = []
   for root, dirs, files in os.walk(base_path):
       if any(f.endswith('.mid') for f in files):
           midi_dirs.append(root)
   return midi_dirs

base_path = "artefacts/applications"
ground_truth_path = f"{base_path}/ground_truth"
test_dirs = find_midi_dirs(base_path)

metric = FrechetMusicDistance(feature_extractor='clamp2', gaussian_estimator='mle', verbose=True)

results = []
for test_dir in test_dirs:
    if test_dir != ground_truth_path:
        score = metric.score(
            reference_path=ground_truth_path,
            test_path=test_dir
        )
        # try to read model and task from path
        task = test_dir.split("/")[-1]
        model = test_dir.split("/")[-2]

        # if task is ground_truth, set model to ground_truth
        if task == "ground_truth":
            model = "ground_truth"


        results.append({
            "model": model,
            "task": task,
            "score": score
            })
        print(f"{test_dir}: {score}")

# output results as json
import json
with open("artefacts/applications/fmd_results.json", "w") as f:
   json.dump(results, f)

