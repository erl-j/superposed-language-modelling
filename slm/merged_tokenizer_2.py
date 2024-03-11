import symusic
import pydash
import numpy as np
import torch
import pretty_midi

# has features over old one.
# supports velocity bins
# drums are now a separate instrument
# instrument classes
# metadata is part of note attributes

class MergedTokenizer2():
    def __init__(self, config):
        self.config = config

        if "separate_drum_pitch" not in self.config:
            self.config["separate_drum_pitch"] = False

        if "use_drum_duration" not in self.config:
            self.config["use_drum_duration"] = True

        self.drum_pitches = list(range(35, 82))

        # exponential tempo bins
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

        # Add meta attributes
        meta_attribute_order = []

        self.vocab = []

        meta_attribute_order.append("tag")
        self.vocab.append("tag:-")
        self.vocab.append("tag:null")
        for tag in self.config["tags"]:
            self.vocab.append("tag:" + tag)    

        if self.config["time_signatures"] is not None:
            time_signatures = sorted(list(set(self.config["time_signatures"])), key=lambda x: (int(x.split("/")[1]), int(x.split("/")[0])))
            meta_attribute_order.append("time_signature")
            self.vocab.append("time_signature:-")
            for time_signature in time_signatures:
                self.vocab.append("time_signature:" + time_signature)

        meta_attribute_order.append("tempo")
        self.vocab.append("tempo:-")
        for tempo_bin in self.tempo_bins:
            self.vocab.append("tempo:" + str(tempo_bin))

        self.meta_attribute_order = meta_attribute_order

        # Add note attributes
        note_attribute_order = []

        if self.config["use_program"]:
            note_attribute_order.append("program")
            self.vocab.append("program:-")
            for program in range(128):
                self.vocab.append("program:" + str(program))

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

        if self.config["use_instrument"]:
            note_attribute_order.append("instrument")
            self.vocab.append("instrument:-")
            for instrument in instrument_class_names:
                self.vocab.append("instrument:" + instrument)

        assert not (self.config["merge_pitch_and_beat"] and self.config["separate_drum_pitch"]), "Can't merge pitch and beat and separate drum pitch at the same time. Not yet implemented."

        if self.config["merge_pitch_and_beat"]:
            note_attribute_order.append("pitch, onset/beat")

            self.vocab.append("pitch, onset/beat:-")
            for pitch in range(self.config["pitch_range"][0], self.config["pitch_range"][1]+1):
                for onset in range(0, self.config["max_beats"]+1):
                    self.vocab.append("pitch, onset/beat:" + str(pitch) + "," + str(onset))
        else:
            note_attribute_order.append("pitch")
            self.vocab.append("pitch:-")
            for pitch in range(self.config["pitch_range"][0], self.config["pitch_range"][1]+1):
                self.vocab.append("pitch:" + str(pitch))
            if self.config["separate_drum_pitch"]:
                for pitch in self.drum_pitches:
                    self.vocab.append("pitch:" + str(pitch) + " (Drums)")
            
            note_attribute_order.append("onset/beat")
            self.vocab.append("onset/beat:-")
            for onset in range(0, self.config["max_beats"]+1):
                self.vocab.append("onset/beat:" + str(onset))

        note_attribute_order.append("onset/tick")
        self.vocab.append("onset/tick:-")
        for onset in range(0, self.config["ticks_per_beat"]):
            self.vocab.append("onset/tick:" + str(onset))

        if self.config["use_offset"]:
            note_attribute_order.append("offset/beat")
            self.vocab.append("offset/beat:-")
            if not self.config["use_drum_duration"]:
                self.vocab.append("offset/beat:none (Drums)")
            for offset in range(0, self.config["max_beats"]+1):
                self.vocab.append("offset/beat:" + str(offset))

            note_attribute_order.append("offset/tick")
            self.vocab.append("offset/tick:-")
            if not self.config["use_drum_duration"]:
                self.vocab.append("offset/tick:none (Drums)")
            for offset in range(0, self.config["ticks_per_beat"]):
                self.vocab.append("offset/tick:" + str(offset))

        note_attribute_order.append("velocity")
        self.vocab.append("velocity:-")
        for velocity in self.velocity_bins:
            self.vocab.append("velocity:" + str(velocity))

        self.note_attribute_order = note_attribute_order + self.meta_attribute_order
        
        self.attributes_per_note=len(self.note_attribute_order)

        self.meta_len = len(self.meta_attribute_order)
        self.total_len = self.attributes_per_note * self.config["max_notes"]

        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}

        self.format_mask = self.get_format_mask()


    def encode(self, sm, tag):
        tokens = self.sm_to_tokens(sm, tag)
        return self.tokens_to_indices(tokens)

    def decode(self, indices):
        tokens = self.indices_to_tokens(indices)
        return self.tokens_to_sm(tokens)

    def tokens_to_indices(self, tokens):
        return [self.token2idx[token] for token in tokens]

    def indices_to_tokens(self, indices):
        return [self.vocab[idx] for idx in indices]
    
    def get_format_mask(self):
        '''
        Returns a format mask for the given tokenization.
        The format mask is a binary matrix of shape (total_len, len(vocab)) where each row corresponds to a token and each column to a valid token.
        '''
        format_mask = np.zeros((self.config["max_notes"] * len(self.note_attribute_order), len(self.vocab)))
        for note_idx in range(self.config["max_notes"]):
            for attr_idx, note_attr in enumerate(self.note_attribute_order):
                for token in self.vocab:
                    if token.startswith(note_attr):
                        format_mask[note_idx * self.attributes_per_note + attr_idx, self.token2idx[token]] = 1
        return format_mask
    
    def constraint_mask(self,tags=None, tempos=None, instruments=None, pitches=None):

        constraint_mask = np.ones((len(self.note_attribute_order), len(self.vocab)), dtype=int)

        for attribute_index, attribute in enumerate(self.note_attribute_order):

            if attribute == "tag":
                if tags is not None:
                    constraint_mask[attribute_index,:] = 0
                    constraint_mask[attribute_index,self.vocab.index("tag:-")] = 1
                    for tag in tags:
                        constraint_mask[attribute_index,self.vocab.index("tag:" + tag)] = 1
            
            if attribute == "tempo":
                if tempos is not None:
                    constraint_mask[attribute_index,:] = 0
                    constraint_mask[attribute_index,self.vocab.index("tempo:-")] = 1
                    for tempo in tempos:
                        constraint_mask[attribute_index,self.vocab.index("tempo:" + str(tempo))] = 1

            if attribute == "instrument":
                if instruments is not None:
                    constraint_mask[attribute_index,:] = 0
                    constraint_mask[attribute_index,self.vocab.index("instrument:-")] = 1
                    for instrument in instruments:
                        constraint_mask[attribute_index,self.vocab.index("instrument:" + instrument)] = 1
                    # allow undefined instrument
        
        # repeat max notes times in the first dimension
        constraint_mask = np.tile(constraint_mask, (self.config["max_notes"], 1))
        return torch.tensor(constraint_mask)
        

    def sm_to_tokens(self, sm, tag):

        sm = sm.resample(tpq=self.config["ticks_per_beat"],min_dur=1)

        # assert right ticks per beat
        assert sm.ticks_per_quarter == self.config["ticks_per_beat"]

        tempo = sm.tempos[0].qpm

        # get notes
        note_encodings = []
        for track in sm.tracks:
            if track.name not in self.config["ignored_track_names"]:
                for note in track.notes:
                    note_encoding = [
                    ]
                    for note_attr in self.note_attribute_order:
                        if note_attr == "program":
                            note_encoding.append("program:" + str(track.program))
                        if note_attr == "instrument":
                            if track.is_drum:
                                note_encoding.append("instrument:Drums")
                            else:
                                note_encoding.append("instrument:" + pretty_midi.program_to_instrument_class(track.program))
                        elif note_attr == "pitch":
                            if "separate_drum_pitch" in self.config and self.config["separate_drum_pitch"] and track.is_drum:
                                if note.pitch < self.drum_pitches[0] or note.pitch > self.drum_pitches[-1]:
                                    # if pitch is out of drum range
                                    # set to closed hi-hat 
                                    note_encoding.append("pitch:42 (Drums)")
                                else:
                                    note_encoding.append("pitch:" + str(note.pitch) + " (Drums)")
                            else:
                                note_encoding.append("pitch:" + str(note.pitch))
                        elif note_attr == "pitch, onset/beat":
                            note_encoding.append("pitch, onset/beat:" + str(note.pitch) + "," + str(note.start // self.config["ticks_per_beat"]))
                        elif note_attr == "onset/beat":
                            note_encoding.append("onset/beat:" + str(note.start // self.config["ticks_per_beat"]))
                        elif note_attr == "onset/tick":                                
                            note_encoding.append("onset/tick:" + str(note.start % self.config["ticks_per_beat"]))
                        elif note_attr == "offset/beat":
                            if track.is_drum and not self.config["use_drum_duration"]:
                                note_encoding.append("offset/beat:" + "none (Drums)")
                            else:
                                note_encoding.append("offset/beat:" + str(note.end // self.config["ticks_per_beat"]))
                        elif note_attr == "offset/tick":
                            if track.is_drum and not self.config["use_drum_duration"]:
                                note_encoding.append("offset/tick:" + "none (Drums)")
                            else:
                                note_encoding.append("offset/tick:" + str(note.end % self.config["ticks_per_beat"]))
                        elif note_attr == "velocity":
                            note_encoding.append("velocity:" + str(self.velocity_to_velocity_bin(note.velocity)))
                        if note_attr == "tempo":
                            note_encoding.append("tempo:" + str(self.tempo_to_tempo_bin(tempo)))
                        if note_attr == "tag":
                            note_encoding.append("tag:" + tag)
                    note_encodings.append(note_encoding)

        # if more notes than max_notes, remove notes
        if len(note_encodings) > self.config["max_notes"]:
            note_encodings = note_encodings[:self.config["max_notes"]]

        # add empty notes up to max_notes
        for i in range(len(note_encodings), self.config["max_notes"]):
            blank_note = [attr + ":-" for attr in self.note_attribute_order]
            note_encodings.append(blank_note)
        
        # shuffle notes
        if self.config["shuffle_notes"]:
            np.random.shuffle(note_encodings)

        # flatten note_encodings
        note_encodings = pydash.flatten(note_encodings)
        return note_encodings
        
    def tokens_to_sm(self, tokens):
      
        # meta = tokens[:self.meta_len]
        notes = tokens
        # print(self.meta_len)
        # notes = tokens[self.meta_len:]
        # for i, meta_attr in enumerate(self.meta_attribute_order):
        #     if meta_attr == "tempo":
        #         assert meta[i].split(":")[0] == "tempo"
        #         qpm = int(tokens[i].split(":")[1])
        #         sm.tempos.append(symusic.Tempo(qpm=qpm, time=0))
        #     elif meta_attr == "time_signature":
        #         assert meta[i].split(":")[0] == "time_signature"
        #         ts = tokens[i].split(":")[1]
        #         ts = ts.split("/")
        #         sm.time_signatures.append(symusic.TimeSignature(numerator=int(ts[0]), denominator=int(ts[1])))
        #     elif meta_attr == "tag":
        #         assert meta[i].split(":")[0] == "tag"
        #         tag = meta[i].split(":")[1]

        # make sublists of note tokens
        notes = [notes[i:i+self.attributes_per_note] for i in range(0, len(notes), self.attributes_per_note)]
        
        note_recs = []

        # parse notes
        for note in notes:

            # if all attributes are "-", skip note
            if any([attr.split(":")[1] == "-" for attr in note]):
                continue

            program = -1
            pitch = None
            onset = None
            onset_tick = None
            offset = None
            offset_tick = None
            velocity = None

            for i, note_attr in enumerate(self.note_attribute_order):                  
                if note_attr == "program":
                    assert note[i].split(":")[0] == "program"  
                    program_str = note[i].split(":")[1]
                    program = int(program_str)

                elif note_attr == "instrument":
                    assert note[i].split(":")[0] == "instrument"  
                    instrument_str = note[i].split(":")[1]
                    if instrument_str == "Drums":
                        # -1 indicates drums
                        program = -1
                    else:
                        program = self.instrument_class_to_program_nrs[instrument_str][0]
                elif note_attr == "pitch":
                    assert note[i].split(":")[0] == "pitch"

                    pitch = int(note[i].split(":")[1].strip(" (Drums)"))
                elif note_attr == "onset/beat":
                    assert note[i].split(":")[0] == "onset/beat"
                    onset = int(note[i].split(":")[1])
                elif note_attr == "onset/tick":
                    assert note[i].split(":")[0] == "onset/tick"
                    onset_tick = int(note[i].split(":")[1])
                elif note_attr == "offset/beat":
                    assert note[i].split(":")[0] == "offset/beat"
                    if note[i].split(":")[1] == "none (Drums)":
                        offset = 0
                    else:
                        offset = int(note[i].split(":")[1])
                elif note_attr == "offset/tick":
                    assert note[i].split(":")[0] == "offset/tick"
                    if note[i].split(":")[1] == "none (Drums)":
                        # make a quarter beat
                        offset_tick = self.config["ticks_per_beat"] // 4
                    else:
                        offset_tick = int(note[i].split(":")[1])
                elif note_attr == "velocity":
                    assert note[i].split(":")[0] == "velocity"
                    velocity = min(int(note[i].split(":")[1]), 127)
                elif note_attr == "pitch, onset/beat":
                    assert note[i].split(":")[0] == "pitch, onset/beat"
                    pitch = int(note[i].split(":")[1].split(",")[0])
                    onset = int(note[i].split(":")[1].split(",")[1])
                elif note_attr == "tempo":
                    assert note[i].split(":")[0] == "tempo"
                    tempo = int(note[i].split(":")[1])
                elif note_attr == "tag":
                    assert note[i].split(":")[0] == "tag"
                    tag = note[i].split(":")[1]
            note_recs.append({
                "pitch": pitch,
                "onset": onset * self.config["ticks_per_beat"] + onset_tick,
                "offset": offset * self.config["ticks_per_beat"] + offset_tick if self.config["use_offset"] else None,
                "velocity": velocity,
                "program": program,
                "tempo": tempo,
                "tag": tag
            })
        if len(note_recs) > 0:
            tempo = note_recs[0]["tempo"]
        else:
            tempo = 120
        sm = symusic.Score()
        # add 4/4 time signature
        sm.time_signatures.append(symusic.TimeSignature(numerator=4, denominator=4,time=0))
        # add tempo
        sm.tempos.append(symusic.Tempo(qpm=tempo, time=0))
        sm = sm.resample(tpq=self.config["ticks_per_beat"],min_dur=1)
        # group by program
        note_recs = pydash.group_by(note_recs, "program")
        for program, notes in note_recs.items():
            if program == -1:
                track = symusic.Track(program=0, name="Drums", is_drum=True)
            else:
                track = symusic.Track(program=program, name=pretty_midi.program_to_instrument_name(program), is_drum=False)
            for note in notes:
                for k,v in note.items():
                    print(k)
                    print(v)
                    print(type(v))

                track.notes.append(symusic.Note(pitch=note["pitch"], time=note["onset"], duration=note["offset"]-note["onset"], velocity=note["velocity"]))
            sm.tracks.append(track)
        return sm
