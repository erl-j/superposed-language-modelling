import symusic
import pydash
import numpy as np
import torch
import pretty_midi

class MergedTokenizer():
    def __init__(self, config):
        self.config = config

        # exponential tempo bins
        self.tempo_bins = np.linspace(np.log(self.config["min_tempo"]), np.log(self.config["max_tempo"]), self.config["n_tempo_bins"])
        self.tempo_bins = np.exp(self.tempo_bins)
        # int
        self.tempo_bins = np.round(self.tempo_bins).astype(int)
        # remove duplicates
        self.tempo_bins = np.unique(self.tempo_bins)
        self.tempo_to_tempo_bin = lambda tempo: int(self.tempo_bins[np.argmin(np.abs(self.tempo_bins - int(tempo)))])

        # Add meta attributes
        meta_attribute_order = []

        self.vocab = []
        meta_attribute_order.append("special")
        self.vocab.append("special:-")
        self.vocab.append("special:sos")

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
            for offset in range(0, self.config["max_beats"]+1):
                self.vocab.append("offset/beat:" + str(offset))

            note_attribute_order.append("offset/tick")
            self.vocab.append("offset/tick:-")
            for offset in range(0, self.config["ticks_per_beat"]):
                self.vocab.append("offset/tick:" + str(offset))

        note_attribute_order.append("velocity")
        self.vocab.append("velocity:-")
        for velocity in range(1, 128):
            self.vocab.append("velocity:" + str(velocity))

        self.note_attribute_order = note_attribute_order
        
        self.attributes_per_note=len(self.note_attribute_order)

        self.meta_len = len(self.meta_attribute_order)
        self.total_len = self.meta_len + self.attributes_per_note * self.config["max_notes"]

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
        # format_mask = torch.zeros(self.total_len, len(self.vocab))
        

        # # go through meta tokens
        # for meta_idx, meta_token in enumerate(self.meta_attribute_order):
        #     for token in self.vocab:
        #         if token.startswith(meta_token) and not token.endswith("-"):
        #             format_mask[meta_idx, self.token2idx[token]] = 1

        format_mask = np.zeros((self.config["max_notes"] * len(self.note_attribute_order), len(self.vocab)))
        for note_idx in range(self.config["max_notes"]):
            for attr_idx, note_attr in enumerate(self.note_attribute_order):
                for token in self.vocab:
                    if token.startswith(note_attr):
                        format_mask[note_idx * self.attributes_per_note + attr_idx, self.token2idx[token]] = 1
        return format_mask

    def sm_to_tokens(self, sm, tag):

        sm = sm.resample(tpq=self.config["ticks_per_beat"],min_dur=1)

        # assert right ticks per beat
        assert sm.ticks_per_quarter == self.config["ticks_per_beat"]

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
                        elif note_attr == "pitch":
                            note_encoding.append("pitch:" + str(note.pitch))
                        elif note_attr == "pitch, onset/beat":
                            note_encoding.append("pitch, onset/beat:" + str(note.pitch) + "," + str(note.start // self.config["ticks_per_beat"]))
                        elif note_attr == "onset/beat":
                            note_encoding.append("onset/beat:" + str(note.start // self.config["ticks_per_beat"]))
                        elif note_attr == "onset/tick":
                            note_encoding.append("onset/tick:" + str(note.start % self.config["ticks_per_beat"]))
                        elif note_attr == "offset/beat":
                            note_encoding.append("offset/beat:" + str(note.end // self.config["ticks_per_beat"]))
                        elif note_attr == "offset/tick":
                            note_encoding.append("offset/tick:" + str(note.end % self.config["ticks_per_beat"]))
                        elif note_attr == "velocity":
                            note_encoding.append("velocity:" + str(note.velocity))
                    note_encodings.append(note_encoding)

        # shuffle notes
        if self.config["shuffle_notes"]:
            np.random.shuffle(note_encodings)

        # if more notes than max_notes, remove notes
        if len(note_encodings) > self.config["max_notes"]:
            note_encodings = note_encodings[:self.config["max_notes"]]

        # add empty notes up to max_notes
        for i in range(len(note_encodings), self.config["max_notes"]):
            blank_note = [attr + ":-" for attr in self.note_attribute_order]
            note_encodings.append(blank_note)

        # flatten note_encodings
        note_encodings = pydash.flatten(note_encodings)
        return note_encodings
        
    def tokens_to_sm(self, tokens):
        sm = symusic.Score()
        sm = sm.resample(tpq=self.config["ticks_per_beat"],min_dur=1)
        meta = tokens[:self.meta_len]
        notes = tokens[self.meta_len:]
        for i, meta_attr in enumerate(self.meta_attribute_order):
            if meta_attr == "tempo":
                assert meta[i].split(":")[0] == "tempo"
                qpm = int(tokens[i].split(":")[1])
                sm.tempos.append(symusic.Tempo(qpm=qpm, time=0))
            elif meta_attr == "time_signature":
                assert meta[i].split(":")[0] == "time_signature"
                ts = tokens[i].split(":")[1]
                ts = ts.split("/")
                sm.time_signatures.append(symusic.TimeSignature(numerator=int(ts[0]), denominator=int(ts[1])))
            elif meta_attr == "tag":
                assert meta[i].split(":")[0] == "tag"
                tag = meta[i].split(":")[1]

        # make sublists of note tokens
        notes = [notes[i:i+self.attributes_per_note] for i in range(0, len(notes), self.attributes_per_note)]
        
        note_recs = []
        # parse notes
        for note in notes:

            # if all attributes are "-", skip note
            if all([attr.split(":")[1] == "-" for attr in note]):
                continue

            program = 0
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
                
                elif note_attr == "pitch":
                    assert note[i].split(":")[0] == "pitch"
                    pitch = int(note[i].split(":")[1])
                elif note_attr == "onset/beat":
                    assert note[i].split(":")[0] == "onset/beat"
                    onset = int(note[i].split(":")[1])
                elif note_attr == "onset/tick":
                    assert note[i].split(":")[0] == "onset/tick"
                    onset_tick = int(note[i].split(":")[1])
                elif note_attr == "offset/beat":
                    assert note[i].split(":")[0] == "offset/beat"
                    offset = int(note[i].split(":")[1])
                elif note_attr == "offset/tick":
                    assert note[i].split(":")[0] == "offset/tick"
                    offset_tick = int(note[i].split(":")[1])
                elif note_attr == "velocity":
                    assert note[i].split(":")[0] == "velocity"
                    velocity = int(note[i].split(":")[1])
                elif note_attr == "pitch, onset/beat":
                    assert note[i].split(":")[0] == "pitch, onset/beat"
                    pitch = int(note[i].split(":")[1].split(",")[0])
                    onset = int(note[i].split(":")[1].split(",")[1])
            note_recs.append({
                "pitch": pitch,
                "onset": onset * self.config["ticks_per_beat"] + onset_tick,
                "offset": offset * self.config["ticks_per_beat"] + offset_tick if self.config["use_offset"] else None,
                "velocity": velocity,
                "program": program if self.config["use_program"] else None,
            })
        # group by program
        note_recs = pydash.group_by(note_recs, "program")
        for program, notes in note_recs.items():
            track = symusic.Track(program=program, name=pretty_midi.program_to_instrument_name(program))
            for note in notes:
                track.notes.append(symusic.Note(pitch=note["pitch"], time=note["onset"], duration=note["offset"]-note["onset"], velocity=note["velocity"]))
            sm.tracks.append(track)
        return sm
