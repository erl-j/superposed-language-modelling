#%%
import librosa

path = "../../artefacts/voice_tones.wav"

sample_rate = 44100
y, sr = librosa.load(path, sr=sample_rate, mono=True)

# trim audio
y, _ = librosa.effects.trim(y, top_db=10)
# segment the audio by pitch

onset = librosa.onset.onset_detect(y=y, sr=sr, units='samples')



split = [ y[onset[i]:onset[i+1]] for i in range(len(onset)-1) ]



# remove short clips
threshold_s = 0.5
split = [ s for s in split if len(s)/sr > threshold_s]

print(len(split))

from IPython.display import Audio, display

for i in range(len(split)):
    display(Audio(split[i], rate=sr))

# %%

# detect pitch
import numpy as np
import matplotlib.pyplot as plt

# detect pitch with pyin

def get_pitch(y,sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    # remove nans
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return None
    # return mean pitch
    # convert to midi pitch
    return int(np.round(librosa.hz_to_midi(np.mean(f0))))


records = [{"audio" : s, "pitch" : get_pitch(s,sr)} for s in split] 

# remove clips with no pitch
records = [r for r in records if r["pitch"] is not None]

#%%
print(records)








# %%
