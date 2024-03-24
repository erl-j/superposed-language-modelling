import symusic

def transpose_sm(sm, semitones):
    new_sm = sm.copy()
    # remove pitches that are out of range
    for track_idx, track in enumerate(new_sm.tracks):
        if not track.is_drum:
            new_notes = []
            for note_idx, note in enumerate(track.notes):
                if note.pitch + semitones < 0 or note.pitch + semitones > 127:
                    continue
                else:
                    note.pitch += semitones
                    new_notes.append(note)
            new_sm.tracks[track_idx].notes = new_notes
    return new_sm