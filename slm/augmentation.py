import symusic

def transpose_sm(midi, semitones):
    # remove pitches that are out of range
    for track in midi.tracks:
        if not track.is_drum:
            track.notes = [
                note for note in track.notes if note.pitch + semitones >= 0 and note.pitch + semitones < 128 
            ]
            track.shift_pitch(semitones)
    return midi