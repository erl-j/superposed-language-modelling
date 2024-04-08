import numpy as np
import torch
import torch.nn.functional as F
import pretty_midi
import matplotlib.pyplot as plt
import seaborn as sns
import os
import IPython.display as ipd
import symusic

def has_drum(sm):
    for track in sm.tracks:
        if track.is_drum and len(track.notes) > 0:
            return True
    return False

def has_harmonic(sm):
    for track in sm.tracks:
        if not track.is_drum and len(track.notes) > 0:
            return True
    return False
            

def get_sm_pitch_range(sm):
    pitches = []
    for track in sm.tracks:
        if not track.is_drum:
            for note in track.notes:
                pitches.append(note.pitch)
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    return min_pitch, max_pitch

def render_directory_with_fluidsynth(midi_dir, audio_dir, overwrite=False):
    os.makedirs(audio_dir, exist_ok=True)
    for midi_file in os.listdir(midi_dir):
        midi_path = midi_dir + "/" + midi_file
        audio_path = audio_dir + "/" + midi_file.replace(".mid", ".wav")
        # open with symusic
        sm = symusic.Score(midi_path)


        # 16 beats in 4/4
        end_4_bars_tick = 16 * sm.ticks_per_quarter
        # assert sm.end() <= end_4_bars_tick
        # crop midi file to 4 bars
        for track in sm.tracks:
            for note in track.notes:
                    note.duration = min(end_4_bars_tick - note.start, note.duration)
        print(f"end of midi file: {sm.end()}")

        tmp_path = midi_path
        tmp_path = tmp_path.replace(".mid", "_cropped.mid")
        tmp_path = tmp_path.replace(midi_dir, audio_dir)
        sm.dump_midi(tmp_path)
        
        # assert sm.end()<40, "midi file is too long"
        # crop to 4 bars

        # midi_path = tmp_path


        if not os.path.exists(audio_path) or overwrite:
            os.system(f"fluidsynth {tmp_path} -F {audio_path}")


def preview(sm, tmp_dir, audio=True):
    # SAMPLE_RATE = 44_100
    os.makedirs(tmp_dir, exist_ok=True)
    midi_path = tmp_dir + "/tmp.mid"
    audio_path = tmp_dir + "/output.wav"
    sm.dump_midi(midi_path)
    pr = piano_roll(sm, tpq=16)
    plt.figure(figsize=(10, 10))
    sns.heatmap(pr, cmap="magma")
    plt.show()

    if audio:
        os.system(f"fluidsynth {midi_path} -F {audio_path}")
        ipd.display(ipd.Audio(audio_path))


def get_scale(scale, range):
    root = scale.split(" ")[0]
    mode = scale.split(" ")[1]

    scales = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "pentatonic": [0, 2, 4, 7, 9],
        "blues": [0, 3, 5, 6, 7, 10],
    }

    if root not in [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]:
        raise ValueError("Root not found")

    if mode not in scales:
        raise ValueError(f"Mode not found, options are {scales.keys()}")

    root_midi = pretty_midi.note_name_to_number(root + "0")

    while root_midi < range[0]:
        root_midi += 12

    midi_notes = []

    octave = 0
    while True:
        for interval in scales[mode]:
            new_note = octave * 12 + root_midi + interval
            if new_note >= range[1]:
                return midi_notes
            else:
                midi_notes.append(new_note)
        octave += 1


def loop_sm(sm, loop_bars, n_loops):
    '''
    4/4 only
    '''
    sm = sm.copy()
    # get resolution
    tpq = sm.ticks_per_quarter

    # get duration of a bar in ticks
    loop_duration = tpq * 4 * loop_bars
    
    for track in sm.tracks:
        new_notes = []
        for loop_idx in range(1,n_loops):
            for note in track.notes:
                note = note.copy()
                note.start = note.start + loop_duration*(loop_idx)
                new_notes.append(note)
        track.notes.extend(new_notes)
    return sm

def piano_roll(sm, tpq):
    sm = sm.copy()
    sm = sm.resample(tpq=tpq, min_dur=0)

    # set all is_drum to False
    for track in sm.tracks:
        track.is_drum = False

    pr = sm.pianoroll(modes=["frame"]).sum(axis=0).sum(axis=0)

    # flip y axis
    pr = pr[::-1]

    return pr

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    # from https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

    Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    return logits