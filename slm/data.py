import torch
import glob.glob
import symusic
import pydash

class MidiDataset(torch.utils.data.Dataset):
    def __init__(self, midi_path, metadata_path):
        self.midi_path = midi_path

augmentation_config = {
    "augmentation_transposition_range":[-7, 7],
    "augmentation_tempo_scale": 0.03,
    "augmentation_velocity_shift": 3,
}
tokenizer_config = {
    "ticks_per_beat":24,
    "pitch_range":[31, 108],
    "max_beats":33,
    "max_notes":128,
    "min_tempo":50,
    "max_tempo":200,
    "n_tempo_bins": 16,
    "time_signatures": None,
    "tags": ["rock","pop"],
    "shuffle_notes": True,
    "use_offset": True,
    "merge_pitch_and_beat":True,
    "use_program": False,
}
