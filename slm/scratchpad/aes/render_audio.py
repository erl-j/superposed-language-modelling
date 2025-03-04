#%%
from symusic import Score, Synthesizer, BuiltInSF3 ,dump_wav
import glob
from tqdm import tqdm
import soundfile as sf
import os

# You could choose a builtin soundfont
# And the following one is the default soundfont if you don't specify it when creating a synthesizer
sf_path = BuiltInSF3.MuseScoreGeneral().path(download=True)

# # sf3 and sf2 are both supported
# sf_path = "path/to/your/soundfont.sf3"

sample_rate = 44100

synth = Synthesizer(
    sf_path = sf_path, # the path to the soundfont
    sample_rate = sample_rate, # the sample rate of the output wave, sample_rate is the default value
)

paths = glob.glob("./artefacts/applications_250e/ground_truth/*.mid",recursive=True)

out_path = "./artefacts/audio"

aba_records = []

for path in tqdm(paths):
    # audio is a 2D numpy array of float32, [channels, time]
    score = Score(path)
    audio = synth.render(score, stereo=True) # stereo is True by default, which means you will get a stereo wave
    os.makedirs(out_path,exist_ok=True)
    
    sf.write(
        out_path + "/"+ path.split("/")[-1].replace(".mid",".wav"),
        audio.T,
        sample_rate,
        # format='WAV',
        # subtype='PCM_16'
    )

    # create a record for the ABA dataset
    # {"path":"a.wav", "start_time":0, "end_time": 5}
    # set start time and end time to 0 end time to the length of the audio
    aba_records.append({"path":path.split("/")[-1].replace(".mid",".wav"), "start_time":0, "end_time": audio.shape[1]/sample_rate})

    # 
    # dump audio artefacts
    # dump_wav(audio, path.replace("ground_truth","audio").replace(".mid",".wav"), sample_rate=44100)
    # display audio with ipd
    # from IPython.display import Audio

    # create
    # Audio(audio, rate=44100)

# save as jsonl file in output dir
import json
with open(out_path + "/aba_records.jsonl","w") as f:
    for record in aba_records:
        f.write(json.dumps(record) + "\n")

# %%


