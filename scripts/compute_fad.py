#%%
import os
import subprocess
experiment_dir = "artefacts/eval_audio/fad_test"

# recursively find all dirs that contain wav files
sys_dirs = []
# get all dir_paths under experimetn dir that contain at least one wav file
for root, dirs, files in os.walk(experiment_dir):
    if any([f.endswith(".wav") for f in files]):
        sys_dirs.append(root)


# remove sys_dirs with "convert" in the name

sys_dirs = [d for d in sys_dirs if "convert" not in d]


reference_a_audio_dir = "artefacts/eval_audio/fad_test/natural"
reference_b_audio_dir = "artefacts/eval_audio/fad_test/natural2"

print(sys_dirs)
#%%

# open output file
for sys_audio_dir in sys_dirs:
    for reference_audio_dir in [reference_a_audio_dir, reference_b_audio_dir]:
        print(f"Computing FAD for {sys_audio_dir}, with reference {reference_audio_dir}")

        # move all midi file to a identical folder structure with midi files


        # compute the FAD and read the output
        # fadtk clap-laion-music reference_audio_dir system_audio_dir
        fad_cmd = f"CUDA_VISIBLE_DEVICES=1 fadtk clap-laion-music {reference_audio_dir} {sys_audio_dir}"
        result = subprocess.run(fad_cmd, shell=True, capture_output=True)
        output = result.stderr.decode("utf-8")
        # get last line

        output = str(output).split("\n")[-2]
        
        with open("fad_results_2.txt", "a") as f:
            # write to file
            # write file
            f.write(f"{output}\n")
        # 