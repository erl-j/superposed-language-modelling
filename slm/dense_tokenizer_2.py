#%%
import numpy as np
import symusic
import pretty_midi
import torch


class DenseTokenizer2():

    def __init__(self, config) -> None:

        self.config = config

        # TODO: turn the instrument related code into external util
        # get instrument_classes
        instrument_class_names = [pretty_midi.program_to_instrument_class(program) for program in range(128)]

        instrument_class_to_program_nrs = {instrument_class: [] for instrument_class in instrument_class_names}
        for program in range(128):
            instrument_class = pretty_midi.program_to_instrument_class(program)
            instrument_class_to_program_nrs[instrument_class].append(program)

        self.instrument_class_to_program_nrs = instrument_class_to_program_nrs

        # get unique, keep order
        instrument_class_names = list(dict.fromkeys(instrument_class_names))
        # add drums
        instrument_class_names.append("Drums")
        self.instrument_class_names = instrument_class_names


        self.tempo_bins = np.linspace(np.log(self.config["min_tempo"]), np.log(self.config["max_tempo"]), self.config["n_tempo_bins"])
        self.tempo_bins = np.exp(self.tempo_bins)
        # int
        self.tempo_bins = np.round(self.tempo_bins).astype(int)
        # remove duplicates
        self.tempo_bins = np.unique(self.tempo_bins)
        self.tempo_to_tempo_bin = lambda tempo: int(self.tempo_bins[np.argmin(np.abs(self.tempo_bins - int(tempo)))])


        if "n_velocity_bins" in self.config and self.config["n_velocity_bins"] is not None:
            self.velocity_bins = np.linspace(0, 128, self.config["n_velocity_bins"])
            self.velocity_bins = np.round(self.velocity_bins).astype(int)
            self.velocity_to_velocity_bin = lambda velocity: int(self.velocity_bins[np.argmin(np.abs(self.velocity_bins - int(velocity)))])
        else:
            self.velocity_bins = range(128)
            self.velocity_to_velocity_bin = lambda velocity: int(velocity)

        vocab = []

        vocab.append("tag:-")
        vocab.append("tag:null")
        for tag in self.config["tags"]:
            vocab.append("tag:" + tag)   

        vocab.append("tempo:-")
        for tempo_bin in self.tempo_bins:
            vocab.append("tempo:" + str(tempo_bin)) 
    
        vocab.append("instrument:-")
        for instrument in instrument_class_names:
            vocab.append("instrument:" + instrument)

        vocab.append("action:-")
        for velocity in self.velocity_bins:
            vocab.append(f"action vel:{velocity}")
        vocab.append("action:hold")

        self.vocab = vocab

        # vocab to index
        self.vocab2idx = {v: i for i, v in enumerate(vocab)}

        self.timesteps = self.config["cells_per_beat"] * self.config["beats_per_bar"] * self.config["n_bars"]

        self.drum_range = [35,82]

        self.n_pitches = self.config["pitch_range"][1] - self.config["pitch_range"][0]
        
        self.n_drums = self.drum_range[1]-self.drum_range[0]

        self.n_voices = self.n_pitches + self.n_drums


    def get_format_mask(self, scale_idxs=None):
        action_idx = [i for i, v in enumerate(self.vocab) if v.startswith("action")]
        action_mh = np.zeros(
            (self.timesteps, self.n_voices, len(self.vocab)), dtype=np.int64
        )
        for action in action_idx:
            action_mh[..., action] = 1

        instrument_idx = [i for i, v in enumerate(self.vocab) if v.startswith("instrument")]
        instrument_mh = np.zeros((self.timesteps, self.n_voices, len(self.vocab)), dtype=np.int64)
        for instrument in instrument_idx:
            instrument_mh[..., instrument] = 1    
        
        tempo_idx = [i for i, v in enumerate(self.vocab) if v.startswith("tempo")]
        tempo_mh = np.zeros((self.timesteps, self.n_voices, len(self.vocab)), dtype=np.int64)
        for tempo in tempo_idx:
            tempo_mh[..., tempo] = 1

        tag_idx = [i for i, v in enumerate(self.vocab) if v.startswith("tag")]
        tag_mh = np.zeros((self.timesteps, self.n_voices, len(self.vocab)), dtype=np.int64)
        for tag in tag_idx:
            tag_mh[..., tag] = 1
        
        return torch.tensor(np.stack([action_mh, instrument_mh, tempo_mh, tag_mh], axis=-2))
    
    # def sanitize(self, x):
    #     # wherever there is hold or no action, set the program to program:-

    #     # one hot encode hold_no_action
    #     # later, wherever there is program, limit action to onsets

    def encode(self, sm, tag_str=None):
        # downsample the score to tick resolution
        sm = sm.resample(tpq=self.config["cells_per_beat"], min_dur=0)

        qpm = sm.tempos[0].qpm
        
        n_timesteps = self.config["cells_per_beat"] * self.config["beats_per_bar"] * self.config["n_bars"]
        
        # number of tracks
        action = np.ones((n_timesteps, self.n_voices), dtype=np.int32) * self.vocab2idx["action:-"]
        instrument = np.ones((n_timesteps, self.n_voices),dtype=np.int32) * self.vocab2idx["instrument:-"]
        tag = np.ones((n_timesteps, self.n_voices), dtype=np.int32) * self.vocab2idx[f"tag:{tag_str}"]
        tempo = np.ones((n_timesteps, self.n_voices), dtype=np.int32) * self.vocab2idx[f"tempo:{str(self.tempo_to_tempo_bin(qpm))}"]

        pitch_to_notes = [[] for _ in range(self.n_voices)]

        for track in sm.tracks:
            if track.name.startswith("Layer"):
                continue
            for note in track.notes:
                if track.is_drum:
                # check that the pitch is within the range
                    if note.pitch < self.drum_range[0] or note.pitch >= self.drum_range[1]:
                        continue
                    else:
                        pitch_to_notes[note.pitch - self.drum_range[0] + self.n_pitches].append(
                            {"note": note, "instrument": "Drums"}
                        )
                else:
                    if note.pitch < self.config["pitch_range"][0] or note.pitch >= self.config["pitch_range"][1]:
                        continue
                    else:
                        pitch_to_notes[note.pitch - self.config["pitch_range"][0]].append(
                            {"note": note, "instrument": pretty_midi.program_to_instrument_class(track.program)}
                        )
                
           
        for voice_idx, notes in enumerate(pitch_to_notes):
            # sort the notes by start time in reverse order
            notes.sort(key=lambda x: x["note"].start)
            for note in notes:
                start_tick = note["note"].start
                if start_tick >= n_timesteps:
                    continue
                end_tick = note["note"].end
                action[start_tick, voice_idx] = self.vocab2idx["action vel:" + str(note["note"].velocity)]
                instrument[start_tick:end_tick, voice_idx] = self.vocab2idx[
                    f"instrument:{note['instrument']}"
                ]
                action[start_tick + 1 : end_tick, voice_idx] = self.vocab2idx[
                    "action:hold"
                ]
                # clear the action after the note ends
                action[end_tick:, voice_idx] = self.vocab2idx["action:-"]

        # stack the action and program
        x = np.stack([action, instrument, tempo, tag], axis=-1)
        return x

    def decode(self, encoded):

        action = encoded[..., 0]
        instrument = encoded[..., 1]
        tempo = encoded[..., 2]
        tag = encoded[..., 3]

        # get tag
        tag_idx = tag[0, 0]
        tag_str = self.vocab[tag_idx]

        # get tempo
        tempo_idx = tempo[0, 0]
        tempo_str = self.vocab[tempo_idx]
        # tempo
        # split the tempo string
        tempo_str = tempo_str.split(":")[-1]
        tempo = int(tempo_str)

        # get pitch notes
        notes = []

        for voice_idx in range(action.shape[1]):
            current_note = None #{"start": None, "duration": None, "velocity": None, "pitch": None, "program": None}
            for time_idx in range(action.shape[0]):
                action_str = self.vocab[action[ time_idx, voice_idx]]
                instrument_str = self.vocab[instrument[time_idx, voice_idx]]
                if instrument_str == "instrument:-":
                    # if the instrument is not set, set it to first instrument
                    instrument_name = pretty_midi.program_to_instrument_class(0)
                else:
                    instrument_name = instrument_str.split(":")[-1]
                if action_str.startswith("action vel"):
                    if current_note is not None:
                        notes.append(current_note)
                    current_note = {"start": time_idx, 
                                    "duration": 1, 
                                    "velocity": min(int(action_str.split(":")[-1]),127), 
                                    "voice_idx": voice_idx, 
                                    "instrument": instrument_name}
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
        instrument_to_notes = {}
        for note in notes:
            instrument = note["instrument"]
            if instrument not in instrument_to_notes:
                instrument_to_notes[instrument] = []
            instrument_to_notes[instrument].append(note)

        sm = symusic.Score()
        sm = sm.resample(tpq=self.config["cells_per_beat"], min_dur=0)

        # add tempo
        sm.tempos.append(symusic.Tempo(qpm=tempo, time=0))
        # add time signature
        sm.time_signatures.append(symusic.TimeSignature(numerator=4, denominator=4, time=0))
        for instrument, notes in instrument_to_notes.items():
            if instrument == "Drums":
                track = symusic.Track(program=0, name="Drums", is_drum=True)
            else:
                track = symusic.Track(program=self.instrument_class_to_program_nrs[instrument][0], name=instrument, is_drum=False)
            for note in notes:
                if instrument == "Drums":
                    track.notes.append(
                        symusic.Note(
                            pitch=note["voice_idx"] + self.drum_range[0] - self.n_pitches,
                            time=note["start"],
                            duration= note["duration"],
                            velocity=note["velocity"],
                        )
                    )
                else:
                    track.notes.append(
                        symusic.Note(
                            pitch=note["voice_idx"] + self.config["pitch_range"][0],
                            time=note["start"],
                            duration= note["duration"],
                            velocity=note["velocity"],
                        )
                    )
            sm.tracks.append(track)
        return sm
            
# %%
