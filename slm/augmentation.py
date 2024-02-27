import symusic

def transpose_sm(midi, semitones):
    midi = symusic.Score(midi)
    for track in midi.tracks:
        track.shift_pitch(semitones)
    # remove pitches that are out of range
    for track in midi.tracks:
        track.notes = [note for note in track.notes if note.pitch >= 0 and note.pitch < 128]
    return midi