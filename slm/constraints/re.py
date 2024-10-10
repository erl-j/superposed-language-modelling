def repitch(e, beat_range, pitch_range, drums, tag="other", tempo=120):
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
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/beat"].issubset(beats) and e[i].a["pitch"].issubset(pitches):
            e[i].a["pitch"] = pitches
    # pad with empty notes
    e += [ec().force_inactive() for e in range(N_EVENTS - len(e))]
    return e


def revelocity(e, beat_range, pitch_range, drums, tag="other", tempo=120):
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
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/beat"].issubset(beats) and e[i].a["pitch"].issubset(pitches):
            e[i].a["velocity"] = ec().a["velocity"]
    # pad with empty notes
    e += [ec().force_inactive() for e in range(N_EVENTS - len(e))]
    return e


def retime(e, beat_range, pitch_range, drums, tag="other", tempo=120):
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
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/beat"].issubset(beats) and e[i].a["pitch"].issubset(pitches):
            e[i].a["onset/beat"] = beats
            e[i].a["offset/beat"] = beats
            e[i].a["onset/tick"] = ec().a["onset/tick"]
            e[i].a["offset/tick"] = ec().a["offset/tick"]

    # pad with empty notes
    e += [ec().force_inactive() for e in range(N_EVENTS - len(e))]
    return e


def reinstrument(e, beat_range, pitch_range, drums, tag="other", tempo=120):
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
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/beat"].issubset(beats) and e[i].a["pitch"].issubset(pitches):
            e[i].a["instrument"] = (
                {"Drums"} if drums else ec().a["instrument"] - {"Drums"}
            )

    # pad with empty notes
    e += [ec().force_inactive() for e in range(N_EVENTS - len(e))]
    return e


def infill(e, beat_range, pitch_range, drums, tag="other", tempo=120):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(ec().ec().quantize_tempo(tempo))}

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
    # e += [ec().force_inactive() for _ in range(40)]

    infill_constraint = {
        "pitch": {
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        }
        | {"-"},
        "onset/beat": {str(r) for r in range(beat_range[0], beat_range[1])} | {"-"},
        "offset/beat": {str(r) for r in range(beat_range[0], beat_range[1])} | {"-"},
        "instrument": ({"Drums"} if drums else ec().a["instruments"] - {"Drums"})
        | {"-"},
        "tag": {tag, "-"},
        "tempo": {str(ec().ec().quantize_tempo(tempo)), "-"},
    }

    # count notes per beat

    # add between notes_to_remove - 10 and notes_to_remove + 10 notes. At least 10 notes
    # lower_bound_notes = max(notes_removed - 10, 10)
    # upper_bound_notes = notes_removed + 10
    # add between 0 and
    # add 3 forced active
    e += [ec().intersect(infill_constraint).force_active() for _ in range(3)]
    if notes_removed > 0:
        e += [
            ec().intersect(infill_constraint).force_active()
            for _ in range(notes_removed // 2)
        ]
        e += [ec().intersect(infill_constraint) for _ in range(notes_removed)]
    #

    print(f"Notes removed: {notes_removed}")

    # # pad with empty notes
    e += [ec().force_inactive() for _ in range(N_EVENTS - len(e))]
    # add 10 empty notes
    # e += [ec().force_inactive() for _ in range(40)]

    return e
