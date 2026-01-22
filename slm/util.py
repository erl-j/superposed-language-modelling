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
from .CONSTANTS import instrument_class_to_selected_program_nr
from matplotlib.patches import Rectangle
import subprocess
import sys
import tinysoundfont


def detail_plot(sm):
    pr_tpq = 12
    tempo = int(sm.tempos[-1].qpm)
    time_sig = f"{sm.time_signatures[-1].numerator}/{sm.time_signatures[-1].denominator}"
    
    # Get instrument names
    instrument_names = [pretty_midi.program_to_instrument_name(track.program) if not track.is_drum else "Drums" for track in sm.tracks]
    
    # Get pianoroll data
    pr = sm.copy().resample(pr_tpq, min_dur=0).pianoroll(modes=["frame"])[0]
    
    # Get unique instruments
    unique_instruments = np.unique(instrument_names)

    # Create a mapping from instrument names to list of track indices
    instrument_to_track_indices = {instrument: [] for instrument in unique_instruments}
    for i, instrument in enumerate(instrument_names):
        instrument_to_track_indices[instrument].append(i)
    
    # We know that it's 4 bars so let's crop it
    loop_ticks = pr_tpq * 4 * 4
    
    # Use a colormap with distinguishable colors
    colors = plt.cm.tab10.colors

    # Get drum track indices
    drum_indices = np.where(np.array(instrument_names) == "Drums")[0]
    has_drums = len(drum_indices) > 0
    
    # Get melodic track indices (all non-drum tracks)
    melodic_indices = [i for i, name in enumerate(instrument_names) if name != "Drums"]
    
    # Get all non-zero pitches from melodic instruments to determine the y-axis range
    all_melodic_pitches = []
    for idx in melodic_indices:
        instrument_pr = pr[idx][:, :loop_ticks]
        pitches = np.where(np.any(instrument_pr > 0, axis=1))[0]
        all_melodic_pitches.extend(pitches)
    
    if all_melodic_pitches:
        # Find the min and max pitches used with padding of ±3
        min_pitch = max(0, min(all_melodic_pitches) - 3)
        max_pitch = min(127, max(all_melodic_pitches) + 3)
    else:
        # Default range if no melodic notes found
        min_pitch = 60 - 24
        max_pitch = 60 + 24
    
    # Set up the figure with two subplots if there are drums
    if has_drums:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(hspace=0.3)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot melodic instruments on the top subplot
    legend_handles = []
    for i, instrument_name in enumerate(unique_instruments):
        if instrument_name == "Drums":
            continue
        
        # Get indices of tracks with this instrument_name
        track_indices = instrument_to_track_indices[instrument_name]
        
        # Sum all channels with the same program number
        instrument_pr = pr[track_indices].sum(axis=0)
        instrument_pr = instrument_pr[:, :loop_ticks]
        
        # Only show non-zero pitches
        non_zero_indices = np.where(instrument_pr > 0)
        
        # Plot with distinct color
        color = colors[i % len(colors)]
        scatter = ax1.scatter(non_zero_indices[1], non_zero_indices[0], 
                             color=color, marker='s', s=10, 
                             label=instrument_name)
        legend_handles.append(scatter)
    
    # Calculate beat positions
    lines_tpq = 1
    
    # Create tick positions for all data points
    time_ticks = np.arange(0, loop_ticks, lines_tpq * pr_tpq)
    
    # Create matching labels
    time_labels = np.arange(0, len(time_ticks))
    
    # Remove default grid
    ax1.grid(False)

    # Add only vertical lines for each beat
    for tick in time_ticks:
        ax1.axvline(x=tick, color='gray', linestyle='--', alpha=0.3)
    
    # Add horizontal lines for each octave
    for octave in range(min_pitch // 12, (max_pitch // 12) + 1):
        pitch = octave * 12
        if min_pitch <= pitch <= max_pitch:
            ax1.axhline(y=pitch, color='gray', linestyle='-', alpha=0.2)
    
    ax1.set_xticks(time_ticks)
    ax1.set_xticklabels(time_labels)
    
    # Set the x and y limits to focus on the actual used pitch range
    ax1.set_xlim(0, loop_ticks)
    ax1.set_ylim(min_pitch, max_pitch)
    
    # Add pitch labels with note names every 12 semitones (C notes)
    pitch_ticks = [pitch for pitch in range(min_pitch - (min_pitch % 12), max_pitch + 12, 12) if min_pitch <= pitch <= max_pitch]
    pitch_labels = [f"C{(pitch // 12) - 1}" for pitch in pitch_ticks]  # C4 is MIDI 60
    ax1.set_yticks(pitch_ticks)
    ax1.set_yticklabels(pitch_labels)
    
    ax1.set_xlabel("Beat")
    ax1.set_ylabel("Pitch")
    ax1.set_title(f"Melodic Instruments - Tempo: {tempo}, Time Signature: {time_sig}")
    
    # Add legend
    ax1.legend(handles=legend_handles, title="Instruments", 
               loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Plot drums if they exist
    if has_drums:
        # Create mapping for drum pitch to name
        pitch_to_drum_name = {
            pitch: pretty_midi.note_number_to_drum_name(pitch) 
            for pitch in range(0, 128)
            if pretty_midi.note_number_to_drum_name(pitch) is not None
        }
        
        # Get drum pianoroll
        drum_pr = np.zeros_like(pr[0])
        for idx in drum_indices:
            drum_pr += pr[idx]
            
        drum_pr = drum_pr[:, :loop_ticks]
        
        # Get non-zero indices in the drum track
        drum_non_zero = np.where(drum_pr > 0)
        
        # Get unique drum pitches used
        unique_drum_pitches = np.unique(drum_non_zero[0])
        
        # Create a mapping of actual pitch to display position
        pitch_to_position = {pitch: i for i, pitch in enumerate(unique_drum_pitches)}
        
        # Map the pitches to display positions
        display_positions = [pitch_to_position[pitch] for pitch in drum_non_zero[0]]
        
        # Plot drum hits as circles (piano roll style)
        ax2.scatter(drum_non_zero[1] + 0.5, display_positions, 
                   color='black', marker='o', s=50, edgecolors='black', linewidths=0.5)
        
        # Add only vertical lines for each beat (same as above)
        for tick in time_ticks:
            ax2.axvline(x=tick, color='gray', linestyle='--', alpha=0.3)
        
        # Set x-axis ticks and limits to match the main plot
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(time_labels)
        ax2.set_xlim(0, loop_ticks)
        
        # Set y-axis labels with drum names
        ax2.set_yticks(range(len(unique_drum_pitches)))
        drum_labels = [f"{pitch} - {pitch_to_drum_name.get(pitch, 'Unknown')}" 
                      for pitch in unique_drum_pitches]
        ax2.set_yticklabels(drum_labels)
        
        # Set y limits with a bit of padding
        ax2.set_ylim(-0.5, len(unique_drum_pitches) - 0.5)
        
        ax2.set_xlabel("Beat")
        ax2.set_ylabel("Drum Type")
        ax2.set_title("Drum Pattern")
    
    plt.tight_layout()
    plt.show()


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
    # print number of notes before
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

def preview_w_player(x_sm):
    """Preview symusic Score with HTML MIDI player"""
    rand_int = np.random.randint(0, 1000000)
    tmp_file_path = "sm_preview_" + str(rand_int) + ".mid"
    x_sm.dump_midi(tmp_file_path)
    ipd.display(MIDIPlayer(tmp_file_path, 400, styler=cifka_advanced, title='My Player'))
    # delete file
    os.remove(tmp_file_path)

def preview_sm(x_sm, soundfont_path=None, sample_rate=44100):
    """
    Preview symusic Score with audio rendering.
    
    Parameters:
    -----------
    x_sm : symusic.Score
        The score to preview
    soundfont_path : str, optional
        Path to soundfont file. If provided, renders audio with tinysoundfont.
        If None, only displays HTML MIDI player.
    sample_rate : int
        Sample rate for audio rendering (default: 44100)
    """
    rand_int = np.random.randint(0, 1000000)
    tmp_file_path = "sm_preview_" + str(rand_int) + ".mid"
    x_sm.dump_midi(tmp_file_path)
    
    # If soundfont path provided, render and display audio with tinysoundfont
    if soundfont_path is not None:
        try:
            # Ensure soundfont exists
            if not os.path.exists(soundfont_path):
                print(f"Soundfont not found at {soundfont_path}")
            else:
                # Determine duration
                try:
                    sm = symusic.Score(tmp_file_path)
                    duration_seconds = sm.end() / sm.ticks_per_quarter / 2
                    duration_seconds = max(duration_seconds, 10.0)
                except:
                    duration_seconds = 10.0
                
                # Initialize the synthesizer
                synth = tinysoundfont.Synth(samplerate=sample_rate)
                sfid = synth.sfload(soundfont_path)
                
                # Clear the synth state
                synth.notes_off()
                synth.sounds_off()
                
                # Flush the synth
                dummy_buffer = synth.generate(4 * sample_rate)
                
                # Create sequencer and load MIDI
                seq = tinysoundfont.Sequencer(synth)
                seq.midi_load(tmp_file_path)
                
                # Generate audio buffer
                buffer_size = int(sample_rate * duration_seconds)
                buffer = synth.generate(buffer_size)
                
                # Convert to numpy array
                block = np.frombuffer(bytes(buffer), dtype=np.float32)
                
                # Reshape to stereo (channels, samples)
                stereo_audio = np.stack([block[::2], block[1::2]])
                
                # Normalize
                if stereo_audio.shape[1] > 0:
                    stereo_audio = stereo_audio / (np.abs(stereo_audio).max() + 1e-6)
                else:
                    stereo_audio = np.zeros((2, sample_rate))
                
                # Save to temporary WAV file and load it
                import scipy.io.wavfile as wavfile
                tmp_audio_path = f"audio_preview_{rand_int}.wav"
                audio_for_display = stereo_audio.T
                # Convert to int16 for WAV
                audio_int16 = (audio_for_display * 32767).astype(np.int16)
                wavfile.write(tmp_audio_path, sample_rate, audio_int16)
                ipd.display(ipd.Audio(tmp_audio_path))
                os.remove(tmp_audio_path)
        except Exception as e:
            print(f"Could not render audio: {e}")
            import traceback
            traceback.print_exc()
    else:
        # No soundfont provided, display HTML MIDI player
        ipd.display(MIDIPlayer(tmp_file_path, 400, styler=cifka_advanced, title='My Player'))
    
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


def crop_sm(sm, n_bars, beats_per_bar):
    sm = sm.copy()
    end_tick = n_bars * sm.ticks_per_quarter * beats_per_bar
    for track_idx in range(len(sm.tracks)):
        track = sm.tracks[track_idx]
        new_notes = []
        for note in track.notes:
            if note.start < end_tick:
                # crop duration
                if note.start + note.duration > end_tick:
                    note.duration = end_tick - note.start
                new_notes.append(note)
        sm.tracks[track_idx].notes = new_notes
    return sm

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

import os
import subprocess
import sys

def download_matrix_soundfont():
    """Download and extract soundfont to assets directory"""
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    soundfont_path = os.path.join(assets_dir, 'MatrixSF_v2.1.5.sf2')
    
    # Check if soundfont already exists
    if os.path.exists(soundfont_path):
        print(f"Soundfont already exists at {soundfont_path}")
        return soundfont_path
    
    # Install gdown if not available
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    # Install unrar if not available
    if subprocess.run(['which', 'unrar'], capture_output=True).returncode != 0:
        print("Installing unrar...")
        subprocess.check_call(['apt-get', 'update'])
        subprocess.check_call(['apt-get', 'install', '-y', 'unrar'])
    
    # Download the soundfont
    print("Downloading soundfont...")
    rar_file = os.path.join(assets_dir, 'MatrixSF SF2 v2.1.5.rar')
    gdown.download('https://drive.google.com/uc?id=1hXoHCSxsq1-oOHcq5ZaEf5i-lgVyKvCy', 
                   rar_file, quiet=False)
    
    # Extract the RAR file
    print("Extracting soundfont...")
    subprocess.check_call(['unrar', 'x', '-y', rar_file], cwd=assets_dir)
    
    # Clean up RAR file
    os.remove(rar_file)
    
    print(f"Soundfont ready at {soundfont_path}")
    return soundfont_path

def get_matrix_soundfont_path():
    """Get path to Matrix soundfont, downloading if necessary"""
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
    soundfont_path = os.path.join(assets_dir, 'MatrixSF_v2.1.5.sf2')
    
    if not os.path.exists(soundfont_path):
        print("Matrix soundfont not found, downloading...")
        return download_matrix_soundfont()
    
    return soundfont_path

def render_midi_with_tinysoundfont(midi_path, sample_rate=44100, duration_seconds=None):
    """Render MIDI file to audio using tinysoundfont with Matrix soundfont"""
    # Get the soundfont path
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
    soundfont_path = os.path.join(assets_dir, 'MatrixSF_v2.1.5.sf2')
    
    # Download soundfont if not available
    if not os.path.exists(soundfont_path):
        print("Matrix soundfont not found, downloading...")
        soundfont_path = download_matrix_soundfont()
    
    # Determine duration if not provided
    if duration_seconds is None:
        try:
            sm = symusic.Score(midi_path)
            duration_seconds = sm.end() / sm.ticks_per_quarter / 2  # Convert ticks to seconds (assuming 120 BPM)
            duration_seconds = max(duration_seconds, 10.0)  # At least 10 seconds
        except:
            duration_seconds = 10.0  # Default fallback
    
    # Initialize the synthesizer
    synth = tinysoundfont.Synth(samplerate=sample_rate)
    sfid = synth.sfload(soundfont_path)
    
    # Clear the synth state
    synth.notes_off()
    synth.sounds_off()
    
    # Flush the synth
    dummy_buffer = synth.generate(4 * sample_rate)
    
    # Create sequencer and load MIDI
    seq = tinysoundfont.Sequencer(synth)
    seq.midi_load(midi_path)
    
    # Generate audio buffer
    buffer_size = int(sample_rate * duration_seconds)
    buffer = synth.generate(buffer_size)
    
    # Convert to numpy array
    block = np.frombuffer(bytes(buffer), dtype=np.float32)
    
    # Reshape to stereo (channels, samples)
    # The buffer is interleaved stereo: left=even, right=odd
    stereo_audio = np.stack([block[::2], block[1::2]])
    
    # Normalize
    if stereo_audio.shape[1] > 0:
        stereo_audio = stereo_audio / (np.abs(stereo_audio).max() + 1e-6)
    else:
        stereo_audio = np.zeros((2, sample_rate))
    
    return stereo_audio, sample_rate

INSTRUMENT_COLORS = {
    "Piano": (0.2, 0.4, 0.8), "Bass": (0.8, 0.2, 0.2), "Guitar": (0.2, 0.7, 0.3),
    "Drums": (0.1, 0.1, 0.1), "Strings": (0.6, 0.3, 0.7), "Brass": (0.9, 0.6, 0.1),
    "Reed": (0.4, 0.7, 0.7), "Organ": (0.5, 0.3, 0.1), "Synth Lead": (0.9, 0.3, 0.6),
    "Synth Pad": (0.4, 0.5, 0.8), "Chromatic Percussion": (0.7, 0.7, 0.2),
}

def plot_piano_roll(sm, highlight=None, fixed_pitch_range=None, drums_only=False, locked_sm=None):
    """
    Plot piano roll with optional highlight box and locked notes visualization.
    locked_sm: A symusic Score containing notes that were constrained/fixed.
    """
    pr_tpq = 12
    tempo = int(sm.tempos[-1].qpm) if sm.tempos else 120
    time_sig = f"{sm.time_signatures[-1].numerator}/{sm.time_signatures[-1].denominator}" if sm.time_signatures else "4/4"
    
    instrument_names = [
        pretty_midi.program_to_instrument_class(t.program) if not t.is_drum else "Drums" 
        for t in sm.tracks
    ]
    
    # Generate piano rolls for the main score
    pr = sm.copy().resample(pr_tpq, min_dur=0).pianoroll(modes=["frame"])[0]
    
    # Generate piano rolls for the locked notes if provided
    locked_pr = None
    if locked_sm:
        try:
            locked_pr = locked_sm.copy().resample(pr_tpq, min_dur=0).pianoroll(modes=["frame"])[0]
        except Exception:
            pass # Fail silently if locked_sm is incompatible or empty

    unique_instruments = np.unique(instrument_names)
    instrument_to_tracks = {inst: [i for i, n in enumerate(instrument_names) if n == inst] for inst in unique_instruments}
    
    loop_ticks = pr_tpq * 4 * 4
    drum_indices = np.where(np.array(instrument_names) == "Drums")[0]
    has_drums = len(drum_indices) > 0
    melodic_indices = [i for i, n in enumerate(instrument_names) if n != "Drums"]
    has_melodic = len(melodic_indices) > 0 and not drums_only
    
    # Determine pitch range
    all_pitches = []
    for idx in melodic_indices:
        if idx < len(pr):
            all_pitches.extend(np.where(np.any(pr[idx][:, :loop_ticks] > 0, axis=1))[0])
    
    if fixed_pitch_range:
        min_pitch, max_pitch = fixed_pitch_range
    elif all_pitches:
        min_pitch, max_pitch = max(0, min(all_pitches) - 3), min(127, max(all_pitches) + 3)
    else:
        min_pitch, max_pitch = 24, 84  # C1 to C6
    
    # Create figure - reduced height for harmonic plots
    if has_drums and has_melodic:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.4, 8), gridspec_kw={'height_ratios': [2, 1]})
        plt.subplots_adjust(hspace=0.3)
    elif has_drums and not has_melodic:
        fig, ax2 = plt.subplots(figsize=(9.4, 4))
        ax1 = None
    else:
        fig, ax1 = plt.subplots(figsize=(8.0, 5))
        ax2 = None
    
    time_ticks = np.arange(0, loop_ticks, pr_tpq)
    
    # Helper to plot notes with per-instrument colors using barh
    def plot_notes_on_ax(ax, track_indices, is_drum, p_roll_source, is_locked_layer=False, drum_pitch_map=None):
        if ax is None: return
        
        fallback_colors = plt.cm.Set2.colors
        target_indices = [idx for idx in track_indices if idx < len(p_roll_source)]
        if not target_indices: return

        if is_drum:
            # Aggregate drums
            combined_pr = sum(p_roll_source[idx] for idx in target_indices)[:, :loop_ticks]
            
            # Find continuous note segments
            for pitch in range(combined_pr.shape[0]):
                row = combined_pr[pitch, :]
                if not np.any(row > 0):
                    continue
                
                # Find note start/end positions
                diff = np.diff(np.concatenate([[0], row > 0, [0]]).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                # Map pitch to ordinal position if needed
                y_pos = drum_pitch_map.get(pitch, pitch) if drum_pitch_map else pitch
                
                for start, end in zip(starts, ends):
                    duration = end - start
                    vel = np.mean(row[start:end])
                    
                    if is_locked_layer:
                        # Desaturated drums (lighten black to gray)
                        brightness = np.clip(vel / 127.0, 0.0, 1.0)
                        base_gray = (brightness, brightness, brightness)
                        # Mix with white to desaturate
                        color = np.clip(np.array(base_gray) + 0.6, 0.0, 1.0)
                        ax.barh(y_pos, duration, left=start, height=0.8,
                               color=color, edgecolor='gray', linewidth=0.3, alpha=0.5)
                    else:
                        # Velocity modulates brightness (grayscale) with power scaling for better perception
                        normalized_vel = np.clip(vel / 127.0, 0.0, 1.0)
                        brightness = normalized_vel  # Floor at 0.15, stronger power scale for more contrast
                        color = (brightness, brightness, brightness)
                        ax.barh(y_pos, duration, left=start, height=0.8,
                               color=color, edgecolor='black', linewidth=0.3)

        else:
            # Melodic - plot per instrument with colors
            for i, inst in enumerate(unique_instruments):
                if inst == "Drums": continue
                
                inst_indices = [x for x in instrument_to_tracks[inst] if x in target_indices]
                if not inst_indices: continue
                
                inst_pr = p_roll_source[inst_indices].sum(axis=0)[:, :loop_ticks]
                
                # Track if we've added a label for this instrument
                labeled = False
                
                # Find continuous note segments for each pitch
                for pitch in range(inst_pr.shape[0]):
                    row = inst_pr[pitch, :]
                    if not np.any(row > 0):
                        continue
                    
                    # Find note start/end positions
                    diff = np.diff(np.concatenate([[0], row > 0, [0]]).astype(int))
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]
                    
                    for start, end in zip(starts, ends):
                        duration = end - start
                        vel = np.mean(row[start:end])
                        
                        if is_locked_layer:
                            # Desaturated instrument color for locked notes
                            base_color = np.array(INSTRUMENT_COLORS.get(inst, fallback_colors[i % len(fallback_colors)]))
                            # Mix heavily with white to desaturate
                            color = np.clip(base_color * 0.3 + 0.7, 0.0, 1.0)
                            ax.barh(pitch, duration, left=start, height=0.8,
                                   color=color, edgecolor='gray', linewidth=0.3, alpha=0.5)
                        else:
                            # Velocity modulates color: lower velocity = lighter (more white mixed in)
                            base_color = np.array(INSTRUMENT_COLORS.get(inst, fallback_colors[i % len(fallback_colors)]))
                            brightness = np.clip((1.0 - vel / 127.0) * 0.5, 0.0, 1.0)
                            color = np.clip(base_color + (1.0 - base_color) * brightness, 0.0, 1.0)
                            
                            # Only label once per instrument
                            label = inst if not labeled else ""
                            labeled = True
                            ax.barh(pitch, duration, left=start, height=0.8,
                                   color=color, edgecolor='black', linewidth=0.3, label=label)

    # Create drum pitch mapping for ordinal y-axis
    drum_pitch_map = None
    unique_drum_pitches = []
    if ax2 and has_drums:
        # Collect all unique drum pitches from main and locked scores
        drum_pr_main = sum(pr[idx] for idx in drum_indices if idx < len(pr))[:, :loop_ticks]
        nz_main = np.where(drum_pr_main > 0)
        unique_pitches = set(nz_main[0])
        
        # Also include locked drums if present
        if locked_pr is not None:
            l_inst_names = [
                pretty_midi.program_to_instrument_class(t.program) if not t.is_drum else "Drums" 
                for t in locked_sm.tracks
            ]
            l_drum_idxs = np.where(np.array(l_inst_names) == "Drums")[0]
            if len(l_drum_idxs) > 0:
                l_drum_pr = sum(locked_pr[idx] for idx in l_drum_idxs if idx < len(locked_pr))[:, :loop_ticks]
                nz_l = np.where(l_drum_pr > 0)
                unique_pitches.update(nz_l[0])
        
        # Sort pitches in drum kit order (low to high is typical)
        unique_drum_pitches = sorted(list(unique_pitches))
        drum_pitch_map = {pitch: i for i, pitch in enumerate(unique_drum_pitches)}
    
    # 1. Plot the Locked (Source) Notes First (if available)
    if locked_pr is not None:
        locked_inst_names = [
            pretty_midi.program_to_instrument_class(t.program) if not t.is_drum else "Drums" 
            for t in locked_sm.tracks
        ]
        locked_drum_indices = np.where(np.array(locked_inst_names) == "Drums")[0]
        locked_melodic_indices = [i for i, n in enumerate(locked_inst_names) if n != "Drums"]
        
        if ax1 and locked_melodic_indices:
            plot_notes_on_ax(ax1, locked_melodic_indices, False, locked_pr, is_locked_layer=True)
        if ax2 and locked_drum_indices:
            plot_notes_on_ax(ax2, locked_drum_indices, True, locked_pr, is_locked_layer=True, drum_pitch_map=drum_pitch_map)
    
    # 2. Plot the Generated (Main) Score on top
    if ax1:
        plot_notes_on_ax(ax1, melodic_indices, False, pr, is_locked_layer=False)
    if ax2:
        plot_notes_on_ax(ax2, drum_indices, True, pr, is_locked_layer=False, drum_pitch_map=drum_pitch_map)

    # Standard Formatting
    if ax1:
        ax1.set_xticks(time_ticks)
        ax1.set_xticklabels(np.arange(len(time_ticks)))
        ax1.set_xlim(0, loop_ticks)
        ax1.set_ylim(min_pitch, max_pitch)
        pitch_ticks = [p for p in range(min_pitch - (min_pitch % 12), max_pitch + 12, 12) if min_pitch <= p <= max_pitch]
        ax1.set_yticks(pitch_ticks)
        ax1.set_yticklabels([f"C{p // 12 - 1}" for p in pitch_ticks])
        ax1.grid(True, which='both', linestyle='--', alpha=0.3, color='gray')
        ax1.set_xlabel("Beat")
        ax1.set_ylabel("Pitch")
        ax1.set_title(f"Harmonic Instruments")
        
        # Fix duplicate labels in legend
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        # Highlight box (Infill)
        if highlight:
            scale = pr_tpq / 24  # tokenizer tpq = 24
            t0, t1 = [t * scale for t in highlight["tick_range"]]
            p0, p1 = highlight["pitch_range"]
            ax1.add_patch(Rectangle((t0, p0), t1 - t0, p1 - p0, lw=1.5, ec="magenta", fc="none", ls="--", alpha=0.8))
            # Add legend entry for edited portion
            from matplotlib.patches import Patch
            by_label["Edited Portion"] = Patch(edgecolor="magenta", facecolor="none", linestyle="--", linewidth=1.5)
        
        if by_label:
            ax1.legend(by_label.values(), by_label.keys(), title="Legend", loc='upper right', bbox_to_anchor=(1.15, 1))
    
    if ax2 and has_drums and unique_drum_pitches:
        ordinal_positions = list(range(len(unique_drum_pitches)))
        
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(np.arange(len(time_ticks)))
        ax2.set_xlim(0, loop_ticks)
        # Set y-ticks at ordinal positions with drum names as labels
        ax2.set_yticks(ordinal_positions)
        ax2.set_yticklabels([pretty_midi.note_number_to_drum_name(p) or str(p) for p in unique_drum_pitches])
        ax2.set_ylim(-0.5, len(unique_drum_pitches) - 0.5)
        ax2.grid(True, which='both', linestyle='--', alpha=0.3, color='gray')
        ax2.set_xlabel("Beat")
        ax2.set_ylabel("Drum Type")
        ax2.set_title("Drum Pattern")
    
    plt.tight_layout()
    return fig