import symusic

def transpose_sm(sm, semitones):
    sm = sm.copy()
    # remove pitches that are out of range
    for track_idx, track in enumerate(sm.tracks):
        if not track.is_drum:
            new_notes = []
            for note_idx, note in enumerate(track.notes):
                new_pitch = note.pitch + semitones
                note.pitch = new_pitch
                new_notes.append(note)
            new_notes = [note for note in new_notes if note.pitch >= 0 and note.pitch <= 127]
            sm.tracks[track_idx].notes = new_notes
    return sm