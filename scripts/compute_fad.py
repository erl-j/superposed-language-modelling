import os
import subprocess
experiment_dir = "artefacts/infilling_test"

reference_audio_dir = experiment_dir + "/natural_audio"

# get all directories under the reference audio directory
sys_dirs = os.listdir(experiment_dir)
# keep only dirs that end with suffix "_audio"
sys_dirs = [d for d in sys_dirs if d.endswith("_audio")]

for sys_dir in sys_dirs:
    sys_audio_dir = experiment_dir + "/" + sys_dir
    # compute the FAD and read the output
    # fadtk clap-laion-music reference_audio_dir system_audio_dir
    fad_cmd = f"CUDA_VISIBLE_DEVICES=4 fadtk clap-laion-music {reference_audio_dir} {sys_audio_dir}"
    result = subprocess.run(fad_cmd, shell=True, capture_output=True)
    # last line of the output is the FAD
    fad = float(result.stdout.decode().split("\n")[-2])
    print(f"FAD for {sys_dir}: {fad}")