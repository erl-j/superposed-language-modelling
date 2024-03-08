import sys
import pytest
from slm import dense_tokenizer_2 
import symusic
from slm.util import piano_roll
import matplotlib.pyplot as plt
import os

def test_midi_load():
    sm = symusic.Score(
        "tests/test_assets/midi/loop_nr_1_n_bars=4.0_start_tick=0_end_tick=16384.mid"
    )
    assert sm.note_num()>1

def test_cycle_consistency():

    N_BARS = 4
    tokenizer_config = {
        "beats_per_bar": 4,
        "cells_per_beat": 4,
        "pitch_range": [20, 101],
        "n_bars": N_BARS,
        "max_notes": 100 * N_BARS,
        "min_tempo": 50,
        "max_tempo": 200,
        "n_tempo_bins": 16,
        "time_signatures": None,
        "tags": ["pop","rock"],
        "ignored_track_names": [f"Layers{i}" for i in range(0, 8)],
    }

    tk = dense_tokenizer_2.DenseTokenizer2(tokenizer_config)
    x = symusic.Score(
        "tests/test_assets/midi/loop_nr_1_n_bars=4.0_start_tick=0_end_tick=16384.mid"
    )

    for track in x.tracks:
        print(track.name)

    y = tk.encode(x, "pop")

    x2 = tk.decode(y)

    y2 = tk.encode(x2, "pop")

    x3 = tk.decode(y2)

    y3 = tk.encode(x3, "pop")

    x4 = tk.decode(y3)

    y4 = tk.encode(x4, "pop")

    for track in x4.tracks:
        print(track.name)

    # save imshow to artefacts
    os.makedirs("artefacts/test_outputs/", exist_ok=True)
    plt.imshow(piano_roll(x))
    plt.savefig("artefacts/test_outputs/piano_roll_a.png")

    plt.imshow(piano_roll(x2))
    plt.savefig("artefacts/test_outputs/piano_roll_b.png")

    plt.imshow(piano_roll(x3))
    plt.savefig("artefacts/test_outputs/piano_roll_c.png")

    plt.imshow(piano_roll(x4))
    plt.savefig("artefacts/test_outputs/piano_roll_d.png")


    assert (y2 == y3).all()

    # # sort y alphabetically
    # y = sorted(y)

    # # sort y2 alphabetically
    # y2 = sorted(y2)

    # # print y and y2 side by side
    # for i in range(len(y)):
    #     print(y[i], y2[i])
    #     assert y[i] == y2[i]

    