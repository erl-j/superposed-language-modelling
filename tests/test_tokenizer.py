import sys
import pytest
from slm import tokenizer
import symusic

def test_midi_load():
    sm = symusic.Score(
        "tests/test_assets/midi/loop_nr_1_n_bars=4.0_start_tick=0_end_tick=16384.mid"
    )
    assert sm.note_num()>1

def test_cycle_consistency():

    N_BARS = 4
    tokenizer_config = {
        "ticks_per_beat":24,
        "pitch_range":[0, 128],
        "max_beats":4*N_BARS,
        "max_notes":100 * N_BARS,
        "min_tempo":50,
        "max_tempo":200,
        "n_tempo_bins": 16,
        "time_signatures": None,
        "tags": ["pop","rock"],
        "shuffle_notes": True,
        "use_offset": True,
        "merge_pitch_and_beat":True,
        "use_program": True,
        "ignored_track_names":[f"Layers{i}" for i in range(0, 8)],
    }

    tk = tokenizer.Tokenizer(tokenizer_config)
    x = symusic.Score(
        "tests/test_assets/midi/loop_nr_1_n_bars=4.0_start_tick=0_end_tick=16384.mid"
    )
    
    y = tk.encode(x, "rock")

    x2 = tk.decode(y)

    y2 = tk.encode(x2, "rock")

    # sort y alphabetically
    y = sorted(y)

    # sort y2 alphabetically
    y2 = sorted(y2)

    # print y and y2 side by side
    for i in range(len(y)):
        print(y[i], y2[i])
        assert y[i] == y2[i]

    