def add(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(ec().quantize_tempo(tempo))}

    beats = set([str(r) for r in range(beat_range[0], beat_range[1])])
    pitches = set(
        [
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )

    notes_before_removal = len(e)

    # remove if in beat range and pitch range
    e = [
        ev
        for ev in e
        if not (ev.a["onset/beat"].issubset(beats) and ev.a["pitch"].issubset(pitches))
    ]

    notes_after_removal = len(e)

    notes_removed = notes_before_removal - notes_after_removal

    # add empty notes
    e += [ec().force_inactive() for e in range(n_events - len(e))]
    return e