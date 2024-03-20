
#%%
import symusic
from util import loop_sm
from tqdm import tqdm
import sys
import os
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

#%%
N_LOOPS = 2
LOOP_BARS = 4
SAMPLE_RATE = 44100
TPQ = 16
VIDEO_FRAME_RATE = 24

# Create a piano roll
midi_path = "./artefacts/shuffle_original.mid"

sm = symusic.Score(midi_path)
# loop the score
sm = loop_sm(sm, loop_bars=LOOP_BARS, n_loops=N_LOOPS)

# write the midi file
os.system("fluidsynth ./artefacts/shuffle_original.mid -F ./artefacts/output_b.wav")
sm.dump_midi("./artefacts/tmp_loop.mid")
os.system("fluidsynth ./artefacts/tmp_loop.mid -F ./artefacts/output.wav")

# Load the audio file
y, sr = librosa.load("./artefacts/output.wav", sr=SAMPLE_RATE)
duration = y.shape[0] / sr
#%%

sm = symusic.Score(midi_path)
piano_roll = sm.resample(tpq=TPQ).pianoroll(modes=["frame"], pitchRange=(21, 108))
piano_roll = piano_roll.sum(axis=(0, 1))

frame_duration = 1/VIDEO_FRAME_RATE

loop_duration = duration / N_LOOPS
frame_idx = 0
os.makedirs("./artefacts/frames", exist_ok=True)


def create_video(piano_roll, duration, frame_duration):
    global X

    fig, ax = plt.subplots()
    im = plt.imshow(piano_roll, aspect="auto", interpolation="none")

    # hide axis
    ax.axis('off')
    # no margin around the image
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    


    im_frames = piano_roll.shape[1]

    vl = ax.axvline(0, ls='-', color='r', lw=3, zorder=10)
    ax.set_xlim(0, im_frames)

    def animate(t):
        print(f"t={t}")
        # X = np.roll(X, +1, axis=0)
        # im.set_array(X)
        # add the play head
        play_head_position = ((t%loop_duration) / loop_duration)*im_frames
        vl.set_xdata([
            play_head_position,
            play_head_position
        ])
        return vl,

    anim = FuncAnimation(fig, animate, frames=np.arange(0, duration, frame_duration), blit=True)

    plt.show()

    return anim


anim = create_video(piano_roll, duration, frame_duration)

# set ffmpeg path
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

# save as other formats
anim.save("test.mp4", fps=VIDEO_FRAME_RATE, extra_args=['-vcodec', 'libx264'])

# combine audio and video
os.system(f"ffmpeg -i test.mp4 -i ./artefacts/output.wav -c:v copy -c:a aac -strict experimental output.mp4")
#%%






# %%
