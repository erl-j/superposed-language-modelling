import symusic
import pydash
import numpy as np
import torch
import pretty_midi
from util import get_scale
import einops
import matplotlib.pyplot as plt
from constraints.core import MusicalEventConstraint
from CONSTANTS import instrument_class_to_selected_program_nr

class Tokenizer():
    def __init__(self, config):
        self.config = config

        if "time_hierarchy" not in self.config:
            self.config["time_hierarchy"] = "beat_tick"

        if "separate_drum_pitch" not in self.config:
            self.config["separate_drum_pitch"] = False

        if "use_drum_duration" not in self.config:
            self.config["use_drum_duration"] = False

        if "use_durations" not in self.config:
            self.config["use_durations"] = False

        if "durations" not in self.config:
            self.config["durations"] = []
        self.drum_pitches = list(range(35, 82))

        if "fold_event_attributes" not in self.config:
            self.config["fold_event_attributes"] = True

        if "midi_types" not in self.config:
            self.config["midi_types"] = []


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

        if len(self.config["midi_types"]) > 0:
            meta_attribute_order.append("midi_type")
            self.vocab.append("midi_type:-")
            for midi_type in self.config["midi_types"]:
                self.vocab.append("midi_type:" + midi_type)

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

 
        note_attribute_order.append("pitch")
        self.vocab.append("pitch:-")
        for pitch in range(self.config["pitch_range"][0], self.config["pitch_range"][1]+1):
            self.vocab.append("pitch:" + str(pitch))
        if self.config["separate_drum_pitch"]:
            for pitch in self.drum_pitches:
                self.vocab.append("pitch:" + str(pitch) + " (Drums)")

        if self.config["time_hierarchy"] == "beat_tick":
            note_attribute_order.append("onset/beat")
            self.vocab.append("onset/beat:-")
            for onset in range(0, self.config["max_beats"]+1):
                self.vocab.append("onset/beat:" + str(onset))

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

            note_attribute_order.append("onset/tick")
            self.vocab.append("onset/tick:-")
            for onset in range(0, self.config["ticks_per_beat"]):
                self.vocab.append("onset/tick:" + str(onset))

        elif self.config["time_hierarchy"] == "tick":
            note_attribute_order.append("onset/global_tick")
            self.vocab.append("onset/global_tick:-")
            for onset in range(0, 1 + (self.config["ticks_per_beat"]*self.config["max_beats"])):
                self.vocab.append("onset/global_tick:" + str(onset))
            if self.config["use_offset"]:
                note_attribute_order.append("offset/global_tick")
                self.vocab.append("offset/global_tick:-")
                if not self.config["use_drum_duration"]:
                    self.vocab.append("offset/global_tick:none (Drums)")
                for offset in range(0, 1 + (self.config["ticks_per_beat"]*self.config["max_beats"])):
                    self.vocab.append("offset/global_tick:" + str(offset))

        if self.config["use_durations"]:
            note_attribute_order.append("duration")
            self.vocab.append("duration:-")
            if not self.config["use_drum_duration"]:
                self.vocab.append("duration:none (Drums)")
            for duration in self.config["durations"]:
                self.vocab.append("duration:" + str(duration))
            self.duration_ticks = np.array([d*4*self.config["ticks_per_beat"] for d in self.config["durations"]])

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

    def mask_to_event_constraint(self, mask):
        # mask has shape (1, n_notes * n_attributes, vocab_size)

        attributes, vocab = mask.shape

        tokenizer = self
        ec = lambda: MusicalEventConstraint(tokenizer)

        event_constraint = ec()
        for attr_idx in range(attributes):
            # get all tokens
            tokens = mask[attr_idx, :].nonzero().squeeze().tolist()
            # if not list make it a list
            if not isinstance(tokens, list):
                tokens = [tokens]
            # get strs
            tokens = [self.vocab[token].split(":")[-1] for token in tokens]
            note_attr = self.note_attribute_order[attr_idx]

            event_constraint.a[note_attr] = set(tokens)
        return event_constraint
    
    def sanitize_mask(self, mask, event_indices):
        mask_type = mask.dtype
        mask_device = mask.device
        if self.config["fold_event_attributes"]:
            mask = einops.rearrange(mask, "b (n a) v -> b n a v", a=len(self.note_attribute_order))
        for event_idx in event_indices:
            event_mask = mask[0, event_idx, :, :]
            ec = self.mask_to_event_constraint(event_mask)
            ec = ec.sanitize_undef()
            mask[0, event_idx, :, :] = self.event_constraint_to_mask(ec)
        if self.config["fold_event_attributes"]:
            mask = einops.rearrange(mask, "b n a v -> b (n a) v")
        return mask.to(mask_type).to(mask_device)

    def event_constraint_to_mask(self, ec):
        mask = torch.zeros((len(self.note_attribute_order), len(self.vocab)))
        for attr_idx, attr in enumerate(self.note_attribute_order):
            # get tokens
            tokens = list(ec.a[attr])
            tokens = [attr + ":" + token for token in tokens]
            token_idxs = [self.token2idx[token] for token in tokens]
            mask[attr_idx, token_idxs] = 1
        return mask
    
    def event_constraints_to_mask(self, ecs):
        event_masks = [self.event_constraint_to_mask(ec) for ec in ecs]
        mask = torch.stack(event_masks, dim=0)[None, ...]
        # reshape 
        if self.config["fold_event_attributes"]:
            mask = einops.rearrange(mask, "b n a v -> b (n a) v")
        return mask
            
    def encode(self, sm, tag, midi_type=None):
        tokens = self.sm_to_tokens(sm, tag, midi_type)
        if self.config["fold_event_attributes"]:
            # output of size (n_notes * n_attributes)
            return self.tokens_to_indices(tokens)
        else:
            token_ids = self.tokens_to_indices(tokens)
            # convert to numpy
            token_ids = np.array(token_ids)
            token_ids = einops.rearrange(token_ids, "(n a) -> n a", a=self.attributes_per_note)
            return token_ids

    def decode(self, indices):
        if self.config["fold_event_attributes"]:
            tokens = self.indices_to_tokens(indices)
            return self.tokens_to_sm(tokens)
        else:
            indices = einops.rearrange(indices, "n a -> (n a)")
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

    def get_syntax_mask(self):
        '''
        Returns a (n_attributes, vocab_size) mask that indicates which tokens are valid for each attribute.
        '''
        syntax_mask = np.zeros((len(self.note_attribute_order), len(self.vocab)))
        for attr_idx, note_attr in enumerate(self.note_attribute_order):
            for token in self.vocab:
                if token.startswith(note_attr):
                    syntax_mask[attr_idx, self.token2idx[token]] = 1
        return syntax_mask

    def get_instruments(self):
        return self.instrument_class_names
     
    def sm_to_tokens(self, sm, tag, midi_type=None):

        if midi_type is not None:
            assert midi_type in self.config["midi_types"]

        sm = sm.resample(tpq=self.config["ticks_per_beat"],min_dur=1)

        # assert right ticks per beat
        assert sm.ticks_per_quarter == self.config["ticks_per_beat"]

        tempo = sm.tempos[0].qpm

        # get notes
        note_encodings = []
        for track in sm.tracks:
            if track.name not in self.config["ignored_track_names"]:
                for note in track.notes:
                    # if note starts before max_tick, skip
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
                                    # set to hihat
                                    note_encoding.append("pitch:42 (Drums)")
                                else:
                                    note_encoding.append("pitch:" + str(note.pitch) + " (Drums)")
                            else:
                                note_encoding.append("pitch:" + str(note.pitch))
                        elif note_attr == "onset/global_tick":
                            note_encoding.append("onset/global_tick:" + str(note.start))
                        elif note_attr == "offset/global_tick":
                            if track.is_drum and not self.config["use_drum_duration"]:
                                note_encoding.append("offset/global_tick:none (Drums)")
                            else:
                                note_encoding.append(
                                    "offset/global_tick:"
                                    + str(
                                        min(
                                            note.end,
                                            self.config["max_beats"]
                                            * self.config["ticks_per_beat"],
                                        )
                                    )
                                )
                        elif note_attr == "duration":
                            if self.config["use_durations"]:
                                if track.is_drum and not self.config["use_drum_duration"]:
                                    note_encoding.append("duration:none (Drums)")
                                else:
                                    duration = note.end - note.start
                                    # quantize to nearest duration value
                                    duration = self.config["durations"][np.argmin(np.abs(self.duration_ticks - duration))]
                                    note_encoding.append("duration:" + str(duration))
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
                        if note_attr == "midi_type":
                            note_encoding.append("midi_type:" + midi_type)
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

        # 
        return note_encodings    
    
    def get_undefined_probs(self, mask, logits):
        """
        Sample which events should be marked as undefined based on their probabilities.
        
        Args:
            mask: Original event mask [batch_size, sequence_length]
            logits: Raw model outputs [batch_size, sequence_length, vocab_size]
            
        Returns:
            Updated mask with sampled undefined events
        """
        # Store original dtype
        dtype = mask.dtype
        
        # Reshape logits to separate notes and attributes
        num_notes = self.config["max_notes"]
        num_attributes = len(self.note_attribute_order)
        logits = einops.rearrange(logits, "b (n a) v -> b n a v", n=num_notes, a=num_attributes)
        
        # Calculate probabilities for each token
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Create undefined token mask
        undefined_tokens = torch.tensor([
            self.token2idx[attr + ":-"] for attr in self.note_attribute_order
        ], device=logits.device)
        
        # Convert to one-hot and expand dimensions
        undefined_one_hot = torch.nn.functional.one_hot(
            undefined_tokens, 
            num_classes=logits.shape[-1]
        ).to(dtype)  # Convert to same dtype as mask
        
        # Calculate probability of undefined for each note
        undefined_probs = (probs * undefined_one_hot.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # Sum over vocabulary
        # undefined_probs = undefined_probs.mean(dim=-1)  # Average over attributes

        # take max over attributes
        undefined_probs = undefined_probs.max(dim=-1).values
        
        # Sample undefined events using random threshold
        random_nums = torch.rand_like(undefined_probs)
        undefined_events = random_nums < undefined_probs
        
        # Create new mask
        new_mask = mask.clone()
        
        # Reshape undefined_events to match the sequence dimension
        undefined_events_flat = einops.repeat(
            undefined_events,
            'b n -> b (n a)',
            a=num_attributes
        )
        
        # For each undefined event, set all its attributes to undefined
        undefined_pattern = undefined_one_hot[0].to(dtype)  # Convert to same dtype as mask
        new_mask[undefined_events_flat] = undefined_pattern     

        # for non undefined events, remove undefined tokens as an option
        # new_mask[~undefined_events_flat] = new_mask[~undefined_events_flat] * (1 - undefined_pattern)   
        
        return new_mask

    def normalize_constraint(self, x1h):
        # first we convert every column to a list of tokens
        x1h = einops.rearrange(x1h, "b (n a) v -> b n a v", n=self.config["max_notes"], a=len(self.note_attribute_order))

        batch, events, attributes, vocab = x1h.shape

        if batch != 1:
            raise ValueError("Can only normalize single batch")

        event_constraints = []
        for ev in range(events):
            event_constraint = {}
            for attr in range(attributes):
                # get all tokens
                tokens = x1h[0, ev, attr, :].nonzero().squeeze().tolist()
                # if not list make it a list
                if not isinstance(tokens, list):
                    tokens = [tokens]
                # get strs
                tokens = [self.vocab[token] for token in tokens]
                event_constraint[self.note_attribute_order[attr]] = tokens
            event_constraints.append(event_constraint)

        # now run the following rules.

        # every offset_tick must be greater than the minimum onset_tick
        for event_idx, ev in enumerate(event_constraints):
            onset_tick_values = [token.split(":")[1] for token in ev["onset/global_tick"]]
            # keep only numeric values
            onset_tick_non_numeric_values = [t for t in onset_tick_values if not t.isnumeric()]
            onset_tick_numeric_values = [int(t) for t in onset_tick_values if t.isnumeric()]

            offset_tick_values = [token.split(":")[1] for token in ev["offset/global_tick"]]
            offset_tick_non_numeric_values = [t for t in offset_tick_values if not t.isnumeric()]
            offset_tick_numeric_values = [int(t) for t in offset_tick_values if t.isnumeric()]

            # find minimum onset tick
            min_onset_tick = min(onset_tick_numeric_values + [10_0000])
            max_offset_tick = max(offset_tick_numeric_values+ [0])

            # remove all offset ticks that are smaller than min onset tick
            offset_tick_numeric_values = [t for t in offset_tick_numeric_values if t >= min_onset_tick]

            # remove all onsets that are larger than max offset
            onset_tick_numeric_values = [t for t in onset_tick_numeric_values if t <= max_offset_tick]

            onset_tick_values = [str(t) for t in onset_tick_numeric_values] + onset_tick_non_numeric_values
            offset_tick_values = [str(t) for t in offset_tick_numeric_values] + offset_tick_non_numeric_values 

            event_constraints[event_idx]["onset/global_tick"] = ["onset/global_tick:" + t for t in onset_tick_values]
            event_constraints[event_idx]["offset/global_tick"] = ["offset/global_tick:" + t for t in offset_tick_values]

        # convert back to x1h

        x1h = torch.zeros((1, self.config["max_notes"], len(self.note_attribute_order), len(self.vocab)), device=x1h.device, dtype=x1h.dtype)

        for event_idx, ev in enumerate(event_constraints):
            for attr_idx, attr in enumerate(self.note_attribute_order):
                for token in ev[attr]:
                    x1h[0, event_idx, attr_idx, self.token2idx[token]] = 1

        x1h = einops.rearrange(x1h, "b n a v -> b (n a) v")

        return x1h

    
    def collapse_undefined_attributes(self, x1h):

        if not self.config["fold_event_attributes"]:
            x1h = einops.rearrange(x1h, "b n a v -> b (n a) v")
        dtype = x1h.dtype

        x1h = x1h.clone()
        undefined_tokens = [attribute + ":-" for attribute in self.note_attribute_order]

        undefined_token_idx = [self.token2idx[token] for token in undefined_tokens]

        undefined_token_1h = torch.nn.functional.one_hot(torch.tensor(undefined_token_idx,device=x1h.device), num_classes=len(self.vocab)).to(dtype)

        x1h = einops.rearrange(x1h, "b (n a) v -> b n a v", n=self.config["max_notes"], a=len(self.note_attribute_order))

        x1h_has_undefined_attribute = (x1h == undefined_token_1h[None,None,...]).all(dim=-1).any(dim=-1)

        all_undef = undefined_token_1h[None,None,...]

        x1h = torch.where(x1h_has_undefined_attribute[...,None,None], all_undef, x1h)


        if self.config["fold_event_attributes"]:
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
                    onset_beat = int(note[i].split(":")[1])
                elif note_attr == "onset/tick":
                    assert note[i].split(":")[0] == "onset/tick"
                    onset_tick = int(note[i].split(":")[1])
                elif note_attr == "offset/beat":
                    assert note[i].split(":")[0] == "offset/beat"
                    if note[i].split(":")[1] == "none (Drums)":
                        offset_beat = -1
                    else:
                        offset_beat = int(note[i].split(":")[1])
                elif note_attr == "offset/tick":
                    assert note[i].split(":")[0] == "offset/tick"
                    if note[i].split(":")[1] == "none (Drums)":
                        # make a quarter beat
                        offset_tick = -1
                    else:
                        offset_tick = int(note[i].split(":")[1])
                elif note_attr == "onset/global_tick":
                    assert note[i].split(":")[0] == "onset/global_tick"
                    onset_tick = int(note[i].split(":")[1])
                elif note_attr == "offset/global_tick":
                    assert note[i].split(":")[0] == "offset/global_tick"
                    if note[i].split(":")[1] == "none (Drums)":
                        offset_tick = -1
                    else:
                        offset_tick = int(note[i].split(":")[1])
                elif note_attr == "duration":
                    assert note[i].split(":")[0] == "duration"
                    # we do not use duration for decoding only for encoding
                    # it is only used as conditioning!
                elif note_attr == "velocity":
                    assert note[i].split(":")[0] == "velocity"
                    velocity = min(int(note[i].split(":")[1]), 127)
                elif note_attr == "pitch, onset/beat":
                    assert note[i].split(":")[0] == "pitch, onset/beat"
                    pitch = int(note[i].split(":")[1].split(",")[0])
                    onset_beat = int(note[i].split(":")[1].split(",")[1])
                elif note_attr == "tempo":
                    assert note[i].split(":")[0] == "tempo"
                    tempo = int(note[i].split(":")[1])
                elif note_attr == "tag":
                    assert note[i].split(":")[0] == "tag"
                    tag = note[i].split(":")[1]
                elif note_attr == "midi_type":
                    assert note[i].split(":")[0] == "midi_type"
                    midi_type = note[i].split(":")[1]

            if self.config["time_hierarchy"] == "beat_tick":
                onset_tick = onset_beat * self.config["ticks_per_beat"] + onset_tick
                if offset == -1:
                    offset_tick  = onset_tick + self.config["ticks_per_beat"] // 8
                    offset_tick = min(offset_tick, self.config["max_beats"] * self.config["ticks_per_beat"])
                else:
                    offset_tick = offset_beat * self.config["ticks_per_beat"] + offset_tick
            elif self.config["time_hierarchy"] == "tick":
                onset_tick = onset_tick
                if offset_tick == -1:
                    offset_tick = onset_tick + self.config["ticks_per_beat"] // 8
                    offset_tick = min(offset_tick, self.config["max_beats"] * self.config["ticks_per_beat"])

            offset_tick = min(offset_tick, self.config["max_beats"] * self.config["ticks_per_beat"])
            note_recs.append({
                "pitch": pitch,
                "onset": onset_tick,
                "offset":  offset_tick,
                "velocity": velocity,
                "program": program,
                "tempo": tempo,
                "tag": tag,
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
                track = symusic.Track(program=program, 
                                      # name is from isntrument class
                                      name= pretty_midi.program_to_instrument_class(program),
                                      is_drum=False)
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

    
       