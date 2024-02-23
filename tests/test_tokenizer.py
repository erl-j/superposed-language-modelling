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
        "use_program": True,
    }

    tk = tokenizer.Tokenizer(tokenizer_config)
    x = symusic.Score(
        "tests/test_assets/midi/loop_nr_1_n_bars=4.0_start_tick=0_end_tick=16384.mid"
    )
    
    y = tk.sm_to_tokens(x, "rock")

    x2 = tk.tokens_to_sm(y)

    y2 = tk.sm_to_tokens(x2, "rock")

    # sort y alphabetically
    y = sorted(y)

    # sort y2 alphabetically
    y2 = sorted(y2)

    # print y and y2 side by side
    for i in range(len(y)):
        print(y[i], y2[i])
        assert y[i] == y2[i]

    