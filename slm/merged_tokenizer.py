import symusic
import pydash
import numpy as np
import torch
import pretty_midi
from util import get_scale
import einops
import matplotlib.pyplot as plt
# has features over old one.
# supports velocity bins
# drums are now a separate instrument
# instrument classes
# metadata is part of note attributes

instrument_class_to_selected_program_nr = {
    "Piano": 1,
    "Chromatic Percussion":12,
    "Organ":21,
    "Guitar":25,
    "Bass":36,
    "Strings":43,
    "Ensemble":54,
    "Brass":61,
    "Reed":72,
    "Pipe":80,
    "Synth Lead":83,
    "Synth Pad":90,
    "Synth Effects":103,
    "Ethnic":108,
    "Percussive":115,
    "Sound Effects":112
}


class MergedTokenizer():
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
    
    def create_mask(self, events):
        mask = np.zeros((len(events) * self.attributes_per_note, len(self.vocab)), dtype=int)
        for event_idx, event in enumerate(events):
            for attr_idx, note_attr in enumerate(self.note_attribute_order):
                for token in event[note_attr]:
                    mask[event_idx * self.attributes_per_note + attr_idx, self.token2idx[f"{note_attr}:{token}"]] = 1
        # multiply with format mask
        mask = mask * self.get_format_mask()
        return torch.tensor(mask).float()

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
    
    def replace_instruments_mask(self, x, instruments_to_remove, instruments_to_add, min_notes_per_instrument=0, total_max_notes=None):
        x = x.clone()
        # to tokens
        if total_max_notes is None:
            total_max_notes = self.config["max_notes"]
        tokens = self.indices_to_tokens(x)
        # fold into note arrays
        notes = [tokens[i:i+self.attributes_per_note] for i in range(0, len(tokens), self.attributes_per_note)]
        # remove notes with undefined attributes
        active_notes = [attr for attr in notes if not any([attr[i].endswith("-") for i in range(len(self.note_attribute_order))])]
        inactive_notes = [attr for attr in notes if any([attr[i].endswith("-") for i in range(len(self.note_attribute_order))])]
        n_dead = self.config["max_notes"] - total_max_notes
        notes = active_notes + inactive_notes[:n_dead]
        keep_notes = []
        for note in notes:
            if any([note[i].split(":")[1] in set(instruments_to_remove) for i in range(len(self.note_attribute_order))]):
                continue
            keep_notes.append(note)
        
        # return to indices
        tokens = pydash.flatten(keep_notes)
        indices = self.tokens_to_indices(tokens)
        # for each instrument to add, create min_notes_per_instrument notes
        keep_1h = torch.nn.functional.one_hot(torch.tensor(indices,device=x.device), num_classes=len(self.vocab)).float()

        # reshape
        keep = einops.rearrange(keep_1h, "(n a) v -> n a v", n=len(keep_notes), a=len(self.note_attribute_order))
        new = []
        for instrument in instruments_to_add:
            note_event = np.ones((self.attributes_per_note, len(self.vocab)))
            # set all attributes undefined values to 0
            for attr_idx, note_attr in enumerate(self.note_attribute_order):
                undef_token_idx = self.token2idx[note_attr + ":-"]
                note_event[attr_idx, undef_token_idx] = 0
            instrument_idx = self.note_attribute_order.index("instrument")
            # get one hot encoding of instrument
            instrument_token_idx = self.token2idx["instrument:" + instrument]
            instrument_1h = torch.nn.functional.one_hot(torch.tensor([instrument_token_idx],device=x.device), num_classes=len(self.vocab)).float()
            note_event[instrument_idx,:] = instrument_1h
            # repeat for min_notes_per_instrument
            note_events = np.repeat(note_event[None,...], min_notes_per_instrument, axis=0)
            new.append(note_events)
        
        new = np.concatenate(new, axis=0)

        # concatenate
        x_1h = np.concatenate([keep, new], axis=0)


        # count missing note events
        missing_note_events = self.config["max_notes"] - x_1h.shape[0]

        optional_notes = torch.ones((missing_note_events, len(self.note_attribute_order), len(self.vocab)))
        # add missing note events
        for note_idx in range(missing_note_events):
            instrument_idx = self.note_attribute_order.index("instrument")
            optional_notes[note_idx, instrument_idx,:] = 0
            for instrument in instruments_to_add:
                # get one hot encoding of instrument
                instrument_token_idx = self.token2idx["instrument:" + instrument]
                instrument_1h = torch.nn.functional.one_hot(torch.tensor(instrument_token_idx,device=x.device), num_classes=len(self.vocab)).float()
                # Shapes are (22, 9, 374) and torch.Size([374])
                # reshape instrument_1h
                optional_notes[note_idx, instrument_idx,:] += instrument_1h
            undef_instrument_token_idx = self.token2idx["instrument:-"]
            undef_instrument_1h = torch.nn.functional.one_hot(torch.tensor(undef_instrument_token_idx,device=x.device), num_classes=len(self.vocab)).float()
            undef_instrument_1h = undef_instrument_1h
            optional_notes[note_idx, instrument_idx,:] += undef_instrument_1h

        # concatenate all
        x_1h = np.concatenate([x_1h, optional_notes], axis=0)

        x_1h = einops.rearrange(x_1h, "n a v -> (n a) v")

        # multiply by format mask
        x_1h = x_1h * self.get_format_mask()

        return torch.tensor(x_1h)




        

            



    
    def replace_mask(self,x, attributes_to_replace):
        x = x.clone()

        # convert to onehot
        x_1h = np.eye(len(self.vocab))[x]

        for attribute in attributes_to_replace:
            attribute_1h = np.zeros((len(self.vocab)))
            for token in self.vocab:
                if token.startswith(attribute+":"):
                    attribute_1h[self.token2idx[token]] = 1
            attribute_idx = self.note_attribute_order.index(attribute)
            # replace all tokens of attribute with attribute_1h
            x_1h[attribute_idx::len(self.note_attribute_order),:] = attribute_1h

        x_1h = x_1h.reshape((self.config["max_notes"], len(self.note_attribute_order), len(self.vocab)))

        # shuffle in the first dimension
        if self.config["shuffle_notes"]:
            # shuffle notes
            np.random.shuffle(x_1h)

        x_1h = x_1h.reshape((self.config["max_notes"] * len(self.note_attribute_order), len(self.vocab)))
        return torch.tensor(x_1h)
    

    # def constraint_to_mask(self, x, constraints):
                    

    def shuffle_notes_mask(self, x, same_onset_times=False):
        x = x.clone()

        note_mask = np.zeros((len(self.note_attribute_order), len(self.vocab)), dtype=int)

        onset_beat_tokens = [token for token in self.vocab if token.startswith("onset/beat")]
        onset_tick_tokens = [token for token in self.vocab if token.startswith("onset/tick")]
        offset_beat_tokens = [token for token in self.vocab if token.startswith("offset/beat")]
        offset_tick_tokens = [token for token in self.vocab if token.startswith("offset/tick")]

        # get token idxs
        onset_beat_idxs = [self.token2idx[token] for token in onset_beat_tokens]
        onset_tick_idxs = [self.token2idx[token] for token in onset_tick_tokens]
        offset_beat_idxs = [self.token2idx[token] for token in offset_beat_tokens]
        offset_tick_idxs = [self.token2idx[token] for token in offset_tick_tokens]

        note_mask[self.note_attribute_order.index("onset/beat"), onset_beat_idxs] = 1
        note_mask[self.note_attribute_order.index("onset/tick"), onset_tick_idxs] = 1
        note_mask[self.note_attribute_order.index("offset/beat"), offset_beat_idxs] = 1
        note_mask[self.note_attribute_order.index("offset/tick"), offset_tick_idxs] = 1

        # repeat max notes times in the first dimension
        note_mask = np.tile(note_mask, (self.config["max_notes"], 1))

        # one hot encode x
        x_1h = np.zeros((len(self.note_attribute_order)*self.config["max_notes"], len(self.vocab)), dtype=int)
        x_1h[np.arange(len(x_1h)), x] = 1

        # add to x and clamp
        x_1h = x_1h + note_mask
        x_1h = np.clip(x_1h, 0, 1)
        return torch.tensor(x_1h)


    def infilling_mask(self, x, beat_range=None, pitches=None, min_notes= None, max_notes=None, mode= "harmonic+drums"):
        '''
        beat_range: tuple of ints, (min_beat, max_beat) : list of strings. If None, defaults to entire beat range.
        pitches : list of strings. If None, defaults to all pitches, including drums.
        '''

        x = x.clone()

        if max_notes is None:
            max_notes = self.config["max_notes"]

        if pitches is None:
            pitches = [token for token in self.vocab if token.startswith("pitch:")]
        
        pitches_set = set(pitches)

        if beat_range is None:
            beat_range = (0, self.config["max_beats"])

        # get tokens
        tokens = self.indices_to_tokens(x)

        # fold into note dictionaries
        notes = [tokens[i:i+self.attributes_per_note] for i in range(0, len(tokens), self.attributes_per_note)]
        # turn into dictionary
        notes = [ {note_attr: [note[i]] for i, note_attr in enumerate(self.note_attribute_order)} for note in notes]
        keep_notes = []
        modify_notes = []

        start = beat_range[0]
        end = beat_range[1]

        start_tick = start * self.config["ticks_per_beat"]
        end_tick = end * self.config["ticks_per_beat"] 
        
        for note_idx, note in enumerate(notes):
            # if any attribute is undefined, keep note
            if any(any([token.endswith("-") for token in note[note_attr]]) for note_attr in self.note_attribute_order):
                continue

            # get onset beat, offset beat, pitch
            onset_tick = int(note["onset/beat"][0].split(":")[1])* self.config["ticks_per_beat"] + int(note["onset/tick"][0].split(":")[1])

            if "none (Drums)" in note["offset/beat"][0]:
                offset_tick = onset_tick + 1
            else:
                offset_tick = int(note["offset/beat"][0].split(":")[1])* self.config["ticks_per_beat"] + int(note["offset/tick"][0].split(":")[1])

            pitch = note["pitch"][0]

            # assert offset >= onset
            assert onset_tick < offset_tick

            if pitch not in pitches_set:
                keep_notes.append(note)
            else:
                # if note is entirely outside of beat range, keep note
                if onset_tick >= end_tick or offset_tick <= start_tick:
                    keep_notes.append(note)

                # if note is entirely inside beat range, ignore note
                elif onset_tick >= start_tick and offset_tick <= end_tick:
                    pass

                # if note goes over beat range, keep note
                elif onset_tick < start_tick and offset_tick > end_tick:
                    keep_notes.append(note)

                # if note is partially inside beat range, modify note
                # note starts before beat range and ends inside beat range
                elif onset_tick < start_tick and start_tick < offset_tick <= end_tick:
                    # open offset
                    note["offset/beat"] = ["offset/beat:" + str(beat) for beat in range(start, end+1)]
                    note["offset/tick"] = ["offset/tick:" + str(tick) for tick in range(self.config["ticks_per_beat"])] 
                    modify_notes.append(note)
                
                # overlapping end
                elif start_tick <= onset_tick < end_tick and offset_tick > end_tick:
                    # open onset
                    note["onset/beat"] = ["onset/beat:" + str(beat) for beat in range(start, end)]
                    note["onset/tick"] = ["onset/tick:" + str(tick) for tick in range(self.config["ticks_per_beat"])] 
                    modify_notes.append(note)

        # convert note_objects to mask
        modify_mask = np.zeros((len(modify_notes) * len(self.note_attribute_order), len(self.vocab)), dtype=int)
        for note_idx, note in enumerate(modify_notes):
            for attr_idx, note_attr in enumerate(self.note_attribute_order):
                for token in note[note_attr]:
                    modify_mask[note_idx * self.attributes_per_note + attr_idx, self.token2idx[token]] = 1
        modify = modify_mask.reshape((len(modify_notes), len(self.note_attribute_order), len(self.vocab)))
             
                    
        keep_mask = np.zeros((len(keep_notes) * len(self.note_attribute_order), len(self.vocab)), dtype=int)
        for note_idx, note in enumerate(keep_notes):
            for attr_idx, note_attr in enumerate(self.note_attribute_order):
                for token in note[note_attr]:
                    keep_mask[note_idx * self.attributes_per_note + attr_idx, self.token2idx[token]] = 1
        keep = keep_mask.reshape((len(keep_notes), len(self.note_attribute_order), len(self.vocab)))

        restriction_mask = self.get_format_mask()[:len(self.note_attribute_order)]
        onset_beat_tokens = [f"onset/beat:{str(beat)}" for beat in range(start, end)] + ["onset/beat:-"]
        onset_beat_mask = np.zeros((len(self.vocab)), dtype=int)
        for token in onset_beat_tokens:
            onset_beat_mask[self.token2idx[token]] = 1
        offset_beat_tokens = [f"offset/beat:{str(beat)}" for beat in range(start, end+1)] + ["offset/beat:-"] + ["offset/beat:none (Drums)"]
        offset_beat_mask = np.zeros((len(self.vocab)), dtype=int)
        for token in offset_beat_tokens:
            offset_beat_mask[self.token2idx[token]] = 1
        pitch_tokens = list(pitches_set) + ["pitch:-"]
        pitch_mask = np.zeros((len(self.vocab)), dtype=int)
        for token in pitch_tokens:
            pitch_mask[self.token2idx[token]] = 1
        restriction_mask[self.note_attribute_order.index("onset/beat"),:] *= onset_beat_mask
        restriction_mask[self.note_attribute_order.index("offset/beat"),:] *= offset_beat_mask
        restriction_mask[self.note_attribute_order.index("pitch"),:] *= pitch_mask


        dead_mask = np.zeros((len(self.note_attribute_order), len(self.vocab)), dtype=int)
        for attr_idx, note_attr in enumerate(self.note_attribute_order):
            dead_mask[attr_idx, self.token2idx[f"{note_attr}:-"]] = 1
        optional_mask = np.clip(restriction_mask + dead_mask, 0, 1)
        forced_mask = np.clip(restriction_mask - dead_mask, 0, 1)

        current_notes = len(keep_notes) + len(modify_notes)

        max_notes = max(current_notes, max_notes)
        min_notes = max(min_notes, current_notes)

        # print(f"Keep notes: {len(keep_notes)}")
        # print(f"Modify notes: {len(modify_notes)}")
        # print(f"Max notes: {max_notes}")
        # print(f"Min notes: {min_notes}")

        dead_notes = self.config["max_notes"] - max_notes
        forced_notes = min_notes - current_notes
        optional_notes = max_notes - current_notes - forced_notes

        # print(f"Forced notes: {forced_notes}")
        # print(f"Optional notes: {optional_notes}")
        # print(f"Dead notes: {dead_notes}")
        # print(f"Sum: {current_notes + forced_notes + optional_notes + dead_notes}")
        
        dead = np.repeat(dead_mask[None,...], dead_notes, axis=0)
        forced = np.repeat(forced_mask[None,...], forced_notes, axis=0)
        optional = np.repeat(optional_mask[None,...], optional_notes, axis=0)


        if mode == "harmonic":
            forced[:,:,self.token2idx["instrument:Drums"]] = 0
            optional[:,:,self.token2idx["instrument:Drums"]] = 0
        
        if mode == "drums":
            for token in self.vocab:
                if token.startswith("instrument:") and not token.endswith("Drums") and not token.endswith("-"):
                    forced[:,:,self.token2idx[token]] = 0
                    optional[:,:,self.token2idx[token]] = 0

        mask = np.concatenate([keep, modify, dead, optional, forced], axis=0)


        # plt.figure(figsize=(10,10))
        # plt.imshow(dead[0,:,:].T, aspect="auto",interpolation="nearest")
        # plt.show()

        # plt.figure(figsize=(10,10))
        # plt.imshow(optional[0,:,:].T, aspect="auto",interpolation="nearest")
        # plt.show()

        # plt.figure(figsize=(10,10))
        # plt.imshow(optional[0,:,:].T+dead[0,:,:].T, aspect="auto",interpolation="nearest")
        # plt.show()

        # print(dead[0].sum(-1))

        # # shuffle in the first dimension
        if self.config["shuffle_notes"]:
            # shuffle notes
            np.random.shuffle(mask)
        
        mask = mask.reshape((self.config["max_notes"] * len(self.note_attribute_order), len(self.vocab)))

        # multiply by format mask
        mask = mask * self.get_format_mask()

        return torch.tensor(mask)
                
    
    def constraint_mask(self,tags=None, tempos=None, instruments=None, pitches=None, onset_beats=None, offset_beats=None, scale="" ,min_notes=1, max_notes=None, min_notes_per_instrument=0):

        if tempos is not None:
            tempos = [self.tempo_to_tempo_bin(tempo) for tempo in tempos] 
        if max_notes is None:
            max_notes = self.config["max_notes"]

        if instruments is not None:
            n_instruments = len(instruments)
            n_instrument_specific_notes = min_notes_per_instrument * n_instruments
        else:
            n_instrument_specific_notes = 0

        min_notes = max(min_notes, n_instrument_specific_notes)

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
                    
            if attribute == "pitch":
                if scale != "":
                    midi_notes = get_scale(scale, self.config["pitch_range"])
                    scale_mask = np.zeros(len(self.vocab))
                    scale_mask[self.vocab.index("pitch:-")] = 1
                    for note in midi_notes:
                        scale_mask[self.vocab.index("pitch:" + str(note))] = 1
                    # allow all drum pitches
                    for token_idx in range(len(self.vocab)):
                        if self.vocab[token_idx].startswith("pitch:") and self.vocab[token_idx].endswith(" (Drums)"):
                            scale_mask[token_idx] = 1
                    constraint_mask[attribute_index,:] = scale_mask
        
        # repeat max notes times in the first dimension
        constraint_mask = np.tile(constraint_mask, (self.config["max_notes"], 1))

        undefined_mask = np.zeros((len(self.note_attribute_order), len(self.vocab)), dtype=int)
        for token in self.vocab:
            if token.endswith("-"):
                undefined_mask[:,self.token2idx[token]] = 1

        # up to min notes, remove undefined attributes
        for i in range(min_notes):
            constraint_mask[i * len(self.note_attribute_order):(i+1) * len(self.note_attribute_order),:] *= (1-undefined_mask)


        if min_notes_per_instrument > 0:
            assert n_instrument_specific_notes <= max_notes, "min_notes_per_instrument * n_instruments must be less than or equal to max_notes"
            for instrument_idx,instrument in enumerate(instruments):
                instrument_one_hot = np.eye(len(self.vocab))[self.token2idx["instrument:" + instrument]]
                for i in range(min_notes_per_instrument):
                    instrument_attribute_index = self.note_attribute_order.index("instrument")
                    constraint_mask[(instrument_idx*min_notes_per_instrument + i) * len(self.note_attribute_order) + instrument_attribute_index,:] = instrument_one_hot


        # after max notes, make all attributes undefined
        for i in range(max_notes, self.config["max_notes"]):
            constraint_mask[i * len(self.note_attribute_order):(i+1) * len(self.note_attribute_order),:] = undefined_mask

        constraint_mask = constraint_mask.reshape(
            (self.config["max_notes"], len(self.note_attribute_order), len(self.vocab))
        )

        # shuffle in the first dimension
        if self.config["shuffle_notes"]:
            # shuffle notes         
            np.random.shuffle(constraint_mask)  

        
        constraint_mask = constraint_mask.reshape(
            (self.config["max_notes"] * len(self.note_attribute_order), len(self.vocab))
        )
        
        # multiply with format mask
        constraint_mask = constraint_mask * self.get_format_mask()

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
                                    # set to cuica
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
    
    def collapse_undefined_attributes(self, x1h):

        dtype = x1h.dtype

        x1h = x1h.clone()
        undefined_tokens = [attribute + ":-" for attribute in self.note_attribute_order]

        undefined_token_idx = [self.token2idx[token] for token in undefined_tokens]

        undefined_token_1h = torch.nn.functional.one_hot(torch.tensor(undefined_token_idx,device=x1h.device), num_classes=len(self.vocab)).to(dtype)

        x1h = einops.rearrange(x1h, "b (n a) v -> b n a v", n=self.config["max_notes"], a=len(self.note_attribute_order))

        x1h_has_undefined_attribute = (x1h == undefined_token_1h[None,None,...]).all(dim=-1).any(dim=-1)

        all_undef = undefined_token_1h[None,None,...]

        x1h = torch.where(x1h_has_undefined_attribute[...,None,None], all_undef, x1h)

        x1h = einops.rearrange(x1h, "b n a v -> b (n a) v")

        return x1h

    def tokens_to_sm(self, tokens):
      
        notes = tokens

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
                        program = instrument_class_to_selected_program_nr[instrument_str]-1#[0]
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
                        offset = -1
                    else:
                        offset = int(note[i].split(":")[1])
                elif note_attr == "offset/tick":
                    assert note[i].split(":")[0] == "offset/tick"
                    if note[i].split(":")[1] == "none (Drums)":
                        # make a quarter beat
                        offset_tick = -1
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
            onset_tick = onset * self.config["ticks_per_beat"] + onset_tick
            if offset == -1:
                offset_tick  = onset_tick + self.config["ticks_per_beat"] // 8
                offset_tick = min(offset_tick, self.config["max_beats"] * self.config["ticks_per_beat"])
            else:
                offset_tick = offset * self.config["ticks_per_beat"] + offset_tick

            offset_tick = min(offset_tick, self.config["max_beats"] * self.config["ticks_per_beat"])
            note_recs.append({
                "pitch": pitch,
                "onset": onset_tick,
                "offset":  offset_tick,
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
                track.notes.append(symusic.Note(pitch=note["pitch"], time=note["onset"], duration=note["offset"]-note["onset"], velocity=note["velocity"]))
            sm.tracks.append(track)

        # assert
        if sm.end() > self.config["max_beats"] * self.config["ticks_per_beat"]:
            # write tokens to file
            with open("error_tokens.txt", "w") as f:
                f.write("\n".join(tokens))
            raise ValueError("End time exceeds maximum beats")
        return sm
