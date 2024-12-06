import torch
import glob
import symusic
import pandas as pd
import itertools
from tqdm import tqdm
import random
from augmentation import transpose_sm


def get_num_notes(sm):
    # for all tracks that don't start with "Layer"
    return sum(
        [len(track.notes) for track in sm.tracks if not track.name.startswith("Layer")]
    )


def random_shift(sm, max_tick, tpq):
    sm = sm.copy()
    sm = sm.resample(tpq)
    shift_ticks = random.randint(0, max_tick)
    for track_idx in range(len(sm.tracks)):
        new_notes = []
        for note_idx in range(len(sm.tracks[track_idx].notes)):
            note = sm.tracks[track_idx].notes[note_idx]
            onset_tick = note.start
            offset_tick = onset_tick + note.duration

            new_onset_tick = (onset_tick + shift_ticks) % max_tick
            new_offset_tick = (offset_tick + shift_ticks) % max_tick

            if new_offset_tick <= new_onset_tick:
                new_offset_tick = max_tick

            new_notes.append(
                symusic.Note(
                    time=new_onset_tick,
                    duration=new_offset_tick - new_onset_tick,
                    pitch=note.pitch,
                    velocity=note.velocity,
                )
            )
        sm.tracks[track_idx].notes = new_notes
    return sm


class MidiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_path,
        genre_list,
        path_filter_fn=None,
        tokenizer=None,
        transposition_range=None,
        min_notes=1,
        max_notes=1e6,
        group_by_source=False,
        sm_filter_fn=None,
        use_random_shift=False,
    ):
        self.tokenizer = tokenizer
        self.records = torch.load(cache_path)
        for i in range(len(self.records)):
            if path_filter_fn is not None:
                self.records[i] = [
                    x for x in self.records[i] if path_filter_fn(x["path"])
                ]
            self.records[i] = [
                {**x, "genre": [g for g in x["genre"] if g in genre_list]}
                for x in self.records[i]
            ]
            # remove midi with less than min_notes
            self.records[i] = [
                x
                for x in self.records[i]
                if min_notes <= get_num_notes(x["midi"]) <= max_notes
            ]
            if sm_filter_fn is not None:
                self.records[i] = [
                    x for x in self.records[i] if sm_filter_fn(x["midi"])
                ]
        self.records = [x for x in self.records if len(x) > 0]
        self.transposition_range = transposition_range
        self.group_by_source = group_by_source
        self.use_random_shift = use_random_shift
        if not self.group_by_source:
            midi_hash = {}
            new_records = []
            for record in self.records:
                n_loops_in_song = len(record)
                for r in record:
                    midi = r["midi"]
                    # hash
                    midi_hash_key = hash(midi)
                    if midi_hash_key not in midi_hash:
                        midi_hash[midi_hash_key] = []
                        new_records.append(
                            {**r, "n_loops_in_parent_song": n_loops_in_song}
                        )
            self.records = new_records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self.group_by_source:
            record = random.choice(self.records[idx])
        else:
            record = self.records[idx]
        midi = record["midi"]
        if self.transposition_range is not None:
            transposition = random.randint(*self.transposition_range)
            midi = transpose_sm(midi, transposition)
        if self.use_random_shift:
            midi = random_shift(
                midi,
                self.tokenizer.config["ticks_per_beat"]
                * self.tokenizer.config["max_beats"],
                tpq=self.tokenizer.config["ticks_per_beat"],
            )

        return {
            "token_ids": torch.tensor(
                self.tokenizer.encode(
                    midi,
                    random.choice(
                        record["genre"] if len(record["genre"]) > 0 else ["other"]
                    ),
                )
            ),
            "n_loops_in_parent_song": torch.tensor([record["n_loops_in_parent_song"]]),
        }


# augmentation_config = {
#     "augmentation_transposition_range":[-7, 7],
#     "augmentation_tempo_scale": 0.03,
#     "augmentation_velocity_shift": 3,
# }
# tokenizer_config = {
#     "ticks_per_beat":24,
#     "pitch_range":[31, 108],
#     "max_beats":33,
#     "max_notes":128,
#     "min_tempo":50,
#     "max_tempo":200,
#     "n_tempo_bins": 16,
#     "time_signatures": None,
#     "tags": ["rock","pop"],
#     "shuffle_notes": True,
#     "use_offset": True,
#     "merge_pitch_and_beat":True,
#     "use_program": False,
# }
