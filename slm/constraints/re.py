# hierarchical
def repitch(
    e,
    ec,
    n_events,
    tick_range,
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

    ticks = set([str(r) for r in range(tick_range[0], tick_range[1])])
    pitches = set(
        [
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/global_tick"].issubset(ticks) and e[i].a["pitch"].issubset(pitches):
            e[i].a["pitch"] = pitches
    # pad with empty notes
    e += [ec().force_inactive() for e in range(n_events- len(e))]
    return e


def revelocity(
    e,
    ec,
    n_events,
    tick_range,
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

    ticks = set([str(r) for r in range(tick_range[0], tick_range[1])])
    pitches = set(
        [
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/global_tick"].issubset(ticks) and e[i].a["pitch"].issubset(pitches):
            e[i].a["velocity"] = ec().a["velocity"]
    # pad with empty notes
    e += [ec().force_inactive() for e in range(n_events - len(e))]
    return e


def retime(
        e,
        ec,
        n_events,
        tick_range,
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

    ticks = set([str(r) for r in range(tick_range[0], tick_range[1])])
    pitches = set(
        [
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/global_tick"].issubset(ticks) and e[i].a["pitch"].issubset(pitches):
            e[i].a["onset/global_tick"] = ticks
            e[i].a["offset/global_tick"] = ticks

    # pad with empty notes
    e += [ec().force_inactive() for e in range(n_events - len(e))]
    return e


def reinstrument(
    e,
    ec,
    n_events,
    tick_range,
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

    ticks = set([str(r) for r in range(tick_range[0], tick_range[1])])
    pitches = set(
        [
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )
    # if in beat range and pitch range, repitch
    for i in range(len(e)):
        if e[i].a["onset/global_tick"].issubset(ticks) and e[i].a["pitch"].issubset(pitches):
            e[i].a["instrument"] = (
                {"Drums"} if drums else ec().a["instrument"] - {"Drums"}
            )

    # pad with empty notes
    e += [ec().force_inactive() for e in range(n_events - len(e))]
    return e


def humanize(
    e,
    ec,
    n_events,
    tick_range,
    pitch_range,
    drums,
    tag,
    tempo):
    # remove empty events

    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(ec().quantize_tempo(tempo))}

    ticks = set([str(r) for r in range(tick_range[0], tick_range[1])])

    pitches = set(
        [
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        ]
    )

    tick_offset = 4
    # if in beat range and pitch range, revelocity
    for i in range(len(e)):
        if e[i].a["onset/global_tick"].issubset(ticks) and e[i].a["pitch"].issubset(pitches):
            e[i].a["velocity"] = ec().a["velocity"]
            # change microtiming (shift up to 2 ticks forward or backward)
            # take tick from set
            onset_tick = int(list(e[i].a["onset/global_tick"])[0])
            # up to 2 ticks forward or backward larger than 0 and less than n_tick
            tick_range = set([str(r) for r in range(onset_tick - tick_offset, onset_tick + tick_offset - 1)])

            if not drums:
                # keep only ticks in valid range
                e[i].a["onset/global_tick"] = tick_range & ticks
                # same for offset
                offset_tick = int(list(e[i].a["offset/global_tick"])[0])
                tick_range = set([str(r) for r in range(offset_tick - tick_offset, offset_tick + tick_offset - 1)])
                e[i].a["offset/global_tick"] = tick_range & ticks

    # pad with empty notes
    e += [ec().force_inactive() for e in range(n_events - len(e))]

    return e



        

def replace(
    e,
    ec,
    n_events,
    tick_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # find instruments that are present
    instruments = set()
    for i in range(len(e)):
        instruments = instruments.union(e[i].a["instrument"])

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(ec().quantize_tempo(tempo))}

    ticks = set([str(r) for r in range(tick_range[0], tick_range[1])])
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
        if not (ev.a["onset/global_tick"].issubset(ticks) and ev.a["pitch"].issubset(pitches))
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
        "onset/global_tick": {str(r) for r in range(tick_range[0], tick_range[1])} | {"-"},
        "offset/global_tick": {str(r) for r in range(tick_range[0], tick_range[1])} | {"-"},
        "instrument": ({"Drums"} if drums else instruments - {"Drums"})
        | {"-"},
        "tag": {tag, "-"},
        "tempo": {str(ec().quantize_tempo(tempo)), "-"},
    }

    # add notes removed with infill constraint
    e += [
        ec().intersect(infill_constraint).force_active()
        for _ in range(notes_removed)
    ]



    # count notes per beat

    # add between notes_to_remove - 10 and notes_to_remove + 10 notes. At least 10 notes
    # lower_bound_notes = max(notes_removed - 10, 10)
    # upper_bound_notes = notes_removed + 10
    # add between 0 and
    # add 3 forced active
    # e += [ec().intersect(infill_constraint).force_active() for _ in range(3)]
    # if notes_removed > 0:
    #     e += [
    #         ec().intersect(infill_constraint).force_active()
    #         for _ in range(notes_removed // 2)
    #     ]
    #     e += [ec().intersect(infill_constraint) for _ in range(notes_removed)]
    #

    print(f"Notes removed: {notes_removed}")

    # # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    # add 10 empty notes
    # e += [ec().force_inactive() for _ in range(40)]

    return e
