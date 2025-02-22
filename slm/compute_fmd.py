from frechet_music_distance import FrechetMusicDistance
import os
import json
import sys
from frechet_music_distance.utils import clear_cache
import glob
from tqdm import tqdm

# clear_cache()

def find_midi_dirs(base_path):
   midi_dirs = []
   for root, dirs, files in os.walk(base_path):
       if any(f.endswith('.mid') for f in files):
           midi_dirs.append(root)
   return midi_dirs

base_path = "artefacts/applications_250e_4x"
ground_truth_path = f"{base_path}/ground_truth"
test_dirs = find_midi_dirs(base_path)


for feature_extractor in ['clamp2']:
    metric = FrechetMusicDistance(feature_extractor=feature_extractor, gaussian_estimator='mle', verbose=True)

    results = []
    individual_results = []
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
                "score": score,
                # "ft": feature_extractor
                })
            print(f"{test_dir}: {score}")

            # # get results for individual examples
            # for midi_file in tqdm(glob.glob(f"{test_dir}/**/*.mid", recursive=True)):
            #     score = metric.score_individual(
            #         reference_path=ground_truth_path,
            #         test_song_path=midi_file
            #     )
            #     individual_results.append({
            #         "filename": midi_file.split("/")[-1],
            #         "model": model,
            #         "task": task,
            #         "score": score,
            #         "midi_path": midi_file
            #     })

    # output results as json
    with open(base_path + f"/fmd_results.json", "w") as f:
        json.dump(results, f)

    # with open(base_path + f"/fmd_individual_results.json", "w") as f:
    #     json.dump(individual_results, f)


# compute fmd for individual examples
