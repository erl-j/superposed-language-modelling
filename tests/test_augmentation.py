from slm.data.data import augmentation
import symusic

def test_transpose():
    # print(tk.vocab)
    sm = symusic.Score(
        "tests/test_assets/midi/loop_nr_1_n_bars=4.0_start_tick=0_end_tick=16384.mid"
    )

    # augment
    sm_1up = augmentation.transpose_sm(sm, 1)

    print(sm_1up.tracks[0].notes[0].pitch)
    print(sm.tracks[0].notes[0].pitch)

    # deaugment
    sm2 = augmentation.transpose_sm(sm_1up, -1)

    # augment again 
    sm_1up2 = augmentation.transpose_sm(sm2, 1)

    # assert that the original and the twice augmented are the same
    for track, track2 in zip(sm_1up.tracks, sm_1up2.tracks):
        if track.is_drum and track2.is_drum:
            continue
        for note,note2 in zip(track.notes, track2.notes):
            assert note.pitch == note2.pitch
            assert note.start == note2.start
            assert note.end == note2.end
            assert note.velocity == note2.velocity

    for track, track2 in zip(sm2.tracks, sm_1up2.tracks):
        if track.is_drum and track2.is_drum:
            continue
        for note,note2 in zip(track.notes, track2.notes):
            assert note.pitch != note2.pitch

    # assert 

    return
        



