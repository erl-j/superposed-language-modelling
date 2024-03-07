#%%
import numpy as np
import symusic
import pretty_midi
import torch


class DenseTokenizer():

    def __init__(self, config) -> None:

        self.config = config

        vocab = []

        vocab.append("program:-")
        for program in range(128):
            vocab.append(f"program:{program}")

        vocab.append("action:-")
        for velocity in range(128):
            vocab.append(f"action vel:{velocity}")
        vocab.append("action:hold")

        self.vocab = vocab

        # vocab to index
        self.vocab2idx = {v: i for i, v in enumerate(vocab)}

        self.timesteps = self.config["cells_per_beat"] * self.config["beats_per_bar"] * self.config["n_bars"]
        self.n_voices = 128*2


    def get_format_mask(self, program_idxs=None, scale_idxs=None):
        action_idx = [i for i, v in enumerate(self.vocab) if v.startswith("action")]

        if program_idxs is not None:
            program_idx = [self.vocab2idx[f"program:{i}"] for i in program_idxs]
            program_idx.append(self.vocab2idx["program:-"])
        else:
            program_idx = [i for i, v in enumerate(self.vocab) if v.startswith("program")]
            # add the program:- token
            

        # multihot
        action_mh = np.zeros((self.timesteps, self.n_voices, len(self.vocab)), dtype=np.int64)
        program_mh = np.zeros((self.timesteps, self.n_voices, len(self.vocab)), dtype=np.int64)

        for action in action_idx:
            action_mh[..., action] = 1

        for program in program_idx:
            program_mh[..., program] = 1    

        return torch.tensor(np.stack([action_mh, program_mh], axis=-2))


    def encode(self, sm, tag=None):
        # downsample the score to tick resolution
        sm = sm.resample(tpq=self.config["cells_per_beat"], min_dur=0)
        
        n_timesteps = self.config["cells_per_beat"] * self.config["beats_per_bar"] * self.config["n_bars"]
        

        # number of tracks
        action = np.ones((n_timesteps, self.n_voices), dtype=np.int32) * self.vocab2idx["action:-"]
        program = np.ones((n_timesteps, self.n_voices),dtype=np.int32) * self.vocab2idx["program:-"]

        pitch_to_notes = [[] for _ in range(self.n_voices)]

        for track in sm.tracks:
            if track.name.startswith("Layer"):
                continue
            for note in track.notes:
                pitch_to_notes[note.pitch + 128 * (track.is_drum)].append({"program": track.program, "note": note})
           
        for voice_idx, notes in enumerate(pitch_to_notes):
            # sort the notes by start time in reverse order
            notes.sort(key=lambda x: x["note"].start)
            for note in notes:
                start_tick = note["note"].start
                if start_tick >= n_timesteps:
                    continue
                end_tick = note["note"].end
                action[start_tick, voice_idx] = self.vocab2idx["action vel:" + str(note["note"].velocity)]
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

        action = encoded[..., 0]
        program = encoded[..., 1]

        # get pitch notes
        notes = []

        for voice_idx in range(action.shape[1]):
            current_note = None #{"start": None, "duration": None, "velocity": None, "pitch": None, "program": None}
            for time_idx in range(action.shape[0]):
                action_str = self.vocab[action[ time_idx, voice_idx]]
                program_str = self.vocab[program[ time_idx, voice_idx]]
                print(action_str, program_str)
                if program_str == "program:-":
                    program_nr = 0
                else:
                    program_nr = int(program_str.split(":")[-1])
                if action_str.startswith("action vel"):
                    if current_note is not None:
                        notes.append(current_note)
                    current_note = {"start": time_idx, "duration": 1, "velocity": int(action_str.split(":")[-1]), "voice_idx": voice_idx, "program": program_nr}
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
                        pitch=note["voice_idx"] % 128,
                        time=note["start"],
                        duration= note["duration"],
                        velocity=note["velocity"],
                    )
                )
                if note["voice_idx"] >= 128:
                    track.is_drum = True
            sm.tracks.append(track)

        return sm
            
# %%
