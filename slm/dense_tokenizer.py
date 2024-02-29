#%%
import numpy as np
import symusic
import pretty_midi

N_BARS = 4
tokenizer_config = {
    "beats_per_bar": 4,
    "cells_per_beat": 24,
    "pitch_range": [0, 128],
    "n_bars": N_BARS,
    "max_notes": 100 * N_BARS,
    "min_tempo": 50,
    "max_tempo": 200,
    "n_tempo_bins": 16,
    "time_signatures": None,
    "tags": ["pop"],
    "shuffle_notes": True,
    "use_offset": True,
    "merge_pitch_and_beat": True,
    "use_program": True,
    "ignored_track_names": [f"Layers{i}" for i in range(0, 8)],
}


class DenseTokenizer():

    def __init__(self, config) -> None:

        self.config = config

        vocab = []

        vocab.append("program:-")
        for program in range(128):
            vocab.append(f"program:{program}")

        vocab.append("action:-")
        for velocity in range(128):
            vocab.append(f"onset vel:{velocity}")
        vocab.append("action:hold")

        self.vocab = vocab

        # vocab to index
        self.vocab2idx = {v: i for i, v in enumerate(vocab)}

    def encode(self, sm):
        # downsample the score to tick resolution
        sm = sm.resample(tpq=24,min_dur=0)
        
        n_timesteps = self.config["cells_per_beat"] * self.config["beats_per_bar"] * self.config["n_bars"]
        n_pitches = 128  

        # number of tracks
        action = np.ones((n_timesteps, n_pitches*2), dtype=np.int32) * self.vocab2idx["action:-"]
        program = np.ones((n_timesteps, n_pitches*2),dtype=np.int32) * self.vocab2idx["program:-"]

        pitch_to_notes = [[] for _ in range(n_pitches*2)]

        for track in sm.tracks:
            if track.name.startswith("Layer"):
                continue
            for note in track.notes:
                pitch_to_notes[note.pitch + n_pitches * (track.is_drum)].append({"program": track.program, "note": note})
           
        for voice_idx, notes in enumerate(pitch_to_notes):
            # sort the notes by start time in reverse order
            notes.sort(key=lambda x: x["note"].start)
            for note in notes:
                start_tick = note["note"].start
                end_tick = note["note"].end
                action[start_tick, voice_idx] = self.vocab2idx["onset vel:" + str(note["note"].velocity)]
                program[start_tick:end_tick, voice_idx] = self.vocab2idx[
                    f"program:{note['program']}"
                ]
                action[start_tick + 1 : end_tick, voice_idx] = self.vocab2idx[
                    "action:hold"
                ]
                # clear the action after the note ends
                action[end_tick:, voice_idx] = self.vocab2idx["action:-"]

        # stack the action and program
        x = np.stack([action, program], axis=-1)
        return x

    def decode(self, encoded):

        n_pitches = 128
        action = encoded[..., 0]
        program = encoded[..., 1]

        # get pitch notes
        notes = []

        print(action.shape)

        for voice_idx in range(action.shape[1]):
            current_note = None #{"start": None, "duration": None, "velocity": None, "pitch": None, "program": None}
            for time_idx in range(action.shape[0]):
                action_str = self.vocab[action[ time_idx, voice_idx]]
                program_str = self.vocab[program[ time_idx, voice_idx]]
                if action_str.startswith("onset"):
                    if current_note is not None:
                        notes.append(current_note)
                    current_note = {"start": time_idx, "duration": 1, "velocity": int(action_str.split(":")[-1]), "voice_idx": voice_idx, "program": int(program_str.split(":")[-1])}
                elif action_str.startswith("action:-"):
                    if current_note is not None:
                        notes.append(current_note)
                        current_note = None
                elif action_str.startswith("action:hold"):
                    if current_note is not None:
                        current_note["duration"] += 1

        # sort the notes by start time
        notes.sort(key=lambda x: x["start"])

        # group notes by program
        program_to_notes = {}
        for note in notes:
            program = note["program"]
            if program not in program_to_notes:
                program_to_notes[program] = []
            program_to_notes[program].append(note)

        sm = symusic.Score()
        sm = sm.resample(tpq=self.config["cells_per_beat"], min_dur=0)
        for program, notes in program_to_notes.items():
            track = symusic.Track(program=program, name=pretty_midi.program_to_instrument_name(program))
            for note in notes:
                track.notes.append(
                    symusic.Note(
                        pitch=note["voice_idx"] % n_pitches,
                        time=note["start"],
                        duration= note["duration"],
                        velocity=note["velocity"],
                    )
                )
            sm.tracks.append(track)

        return sm
            