import numpy as np
import torch
import torch.nn.functional as F
import pretty_midi
import matplotlib.pyplot as plt
import seaborn as sns
import os
import IPython.display as ipd
import symusic
from midi_player import MIDIPlayer
from midi_player.stylers import basic, cifka_advanced
import IPython.display as ipd
import math
import glob
from tqdm import tqdm
from constants import instrument_class_to_selected_program_nr

def sm_set_track_order(sm):
    track_order = ["Drums"] + list(instrument_class_to_selected_program_nr.keys())
    sm = sm.copy()
    # check out current instrument classes
    existing_instrument_classes = set([pretty_midi.program_to_instrument_class(track.program) for track in sm.tracks])
    if any([t.is_drum for t in sm.tracks]):
        existing_instrument_classes.add("Drums")
    
    # add missing instrument classes
    for instrument_class in track_order:
        if instrument_class not in existing_instrument_classes:
            if instrument_class == "Drums":
                sm.tracks.append(symusic.Track(is_drum=True, program=0))
            else:
                sm.tracks.append(symusic.Track(program=instrument_class_to_selected_program_nr[instrument_class], is_drum=False))
    # sort tracks by instrument class
    # sort tracks by is drum
    sm.tracks = sorted(sm.tracks, key=lambda x: x.is_drum)
    sm.tracks = sorted(sm.tracks, key=lambda x: track_order.index(pretty_midi.program_to_instrument_class(x.program)))
    return sm

def sm_fix_overlap_notes(sm):
    sm = sm.copy()
    for track_idx in range(len(sm.tracks)):
        track = sm.tracks[track_idx]
        notes = track.notes
        # sort notes by start
        notes = sorted(notes,key=lambda x: x.start)
        new_notes = []
        for pitch in range(128):
            new_pitch_notes = []
            pitch_notes = [note for note in notes if note.pitch == pitch]
            if len(pitch_notes) > 0:
                new_pitch_notes.append(pitch_notes[0])
                for i in range(len(pitch_notes)-1):
                    if pitch_notes[i].start == pitch_notes[i+1].start:
                        continue
                    elif pitch_notes[i].end > pitch_notes[i+1].start:
                        pitch_notes[i].duration = pitch_notes[i+1].start - pitch_notes[i].start
                    new_pitch_notes.append(pitch_notes[i+1])
            new_notes.extend(new_pitch_notes)
        sm.tracks[track_idx].notes = new_notes
    return sm

def sm_reduce_dynamics(sm, factor):
    sm = sm.copy()
    for track in sm.tracks:
        for note in track.notes:
            note.velocity = int(note.velocity * factor)
    return sm

def load_merged_models(pattern, model_class):
    print("Instantiating merged model")

    ckpt_paths = glob.glob(pattern,recursive=True)

    models = [model_class.load_from_checkpoint(ckpt_path, map_location="cpu") for ckpt_path in ckpt_paths]

    # make a copy of the last model
    merged_model = model_class.load_from_checkpoint(ckpt_paths[-1], map_location="cpu")

    # set merged model to all zero
    for p in merged_model.parameters():
        p.data = torch.zeros_like(p.data)

    total_params = sum(1 for _ in merged_model.parameters())
    progress_bar = tqdm(total=total_params, desc="Merging parameters")

    # print mean of the model prior to merging
    mean_params = [p.data.mean() for p in merged_model.parameters()]
    print(f"Mean of the model prior to merging: {mean_params}")

    for model in models:
        for p, p_merged in zip(model.parameters(), merged_model.parameters()):
            p_merged.data += p.data/len(models)
            progress_bar.update(1)

    # print mean of the model after merging
    mean_params = [p.data.mean() for p in merged_model.parameters()]
    print(f"Mean of the model after merging: {mean_params}")
    
    progress_bar.close()
    return merged_model

def top_p_probs(probs, top_p=0.0):
    # inputs are probs with shape [N, V]
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_indices = sorted_indices.argsort(-1)  # Corrected line
    probs = torch.gather(sorted_probs, 1, sorted_indices)
    # renormalize
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs

def preview_sm(x_sm):
    # rand int
    rand_int = np.random.randint(0, 1000000)
    tmp_file_path = "sm_preview_" + str(rand_int) + ".mid"
    x_sm.dump_midi(tmp_file_path)
    ipd.display(MIDIPlayer(tmp_file_path, 500, styler=cifka_advanced, title='My Player'))
    # delete file
    os.remove(tmp_file_path)


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
        "kaweco":[0,1,5,7]
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

def piano_roll(sm, tpq, include_drums):
    sm = sm.copy()
    sm = sm.resample(tpq=tpq, min_dur=0)

    # if include drums, set all tracks to not be drums
    if include_drums:
        for track in sm.tracks:
            track.is_drum = False
    else:
        # remove drum tracks
        sm.tracks = [track for track in sm.tracks if not track.is_drum]

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