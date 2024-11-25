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
            e[i].a["duration"] = ec().a["duration"]

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
    # if empty every instrument is present
    if len(instruments) == 0:
        instruments = ec().a["instrument"]

    # non drum events
    non_drum_events = [ev for ev in e if "Drums" not in ev.a["instrument"]]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(ec().quantize_tempo(tempo))}

    ticks = set([str(r) for r in range(tick_range[0], tick_range[1])])
    pitches = set(
        [
           f"{r} (Drums)" if drums else f"{r}"  for r in range(pitch_range[0], pitch_range[1])
        ]
    )

    notes_before_removal = len(e)

    infill_region_pitches = pitch_range[1] - pitch_range[0]

    # remove if in beat range and pitch range
    e = [
        ev
        for ev in e
        if not (ev.a["onset/global_tick"].issubset(ticks) and ev.a["pitch"].issubset(pitches))
    ]
    notes_after_removal = len(e)
    notes_removed = notes_before_removal - notes_after_removal

    infill_region_ticks = tick_range[1] - tick_range[0]
    infill_region_bars = infill_region_ticks / 4 * ec().tokenizer.config["ticks_per_beat"]
    # get valid durations
    valid_durations = set()
    for d in ec().tokenizer.config["durations"]:
        if d <= infill_region_bars:
            valid_durations.add(str(d))

    print(f"Valid durations: {valid_durations}")

    valid_onsets = {str(r) for r in range(tick_range[0], tick_range[1])} 
    valid_offsets = {"none (Drums)"} if drums else {str(r) for r in range(tick_range[0]+4, tick_range[1]+1)}
    valid_pitches = pitches
    valid_durations = {"none (Drums)"} if drums else valid_durations
    
    infill_constraint = {
        "pitch": valid_pitches | {"-"},
        "onset/global_tick": valid_onsets | {"-"},
        "offset/global_tick":  valid_offsets | {"-"},
        "instrument": ({"Drums"} if drums else instruments - {"Drums"}) | {"-"},
        # "duration": valid_durations  | {"-"},
    }

    # add notes removed with infill constraint
    # e += [
    #     ec().intersect(infill_constraint).force_active()
    #     for _ in range(1)
    # ]

    # e += [
    #     ec().intersect(infill_constraint).force_active() for _ in range(notes_removed)
    # ]

    e += [ec().intersect(infill_constraint).force_active()  for _ in range(notes_removed)]
    # add one forced note
    # e += [ec().intersect(infill_constraint).force_active() for _ in range(1)]
    # # add 50 optional notes 
    # e += [ec().intersect(infill_constraint) for _ in range(n_events - len(e)-50)]
    # pitch time box size
    pitch_time_box_notes = int( infill_region_bars * infill_region_pitches / 800)


    # # pad with empty notes
    e += [ec().force_inactive() for e in range(n_events - len(e))]

    # set tag
    e = [ev.intersect({"tag":{f"{tag}","-"}}) for ev in e]

    # set tempo

    e = [ e.intersect(ec().tempo_constraint(tempo)) for e in e]

    return e


# def replace(
#     e,
#     ec,
#     n_events,
#     tick_range,
#     pitch_range,
#     drums,
#     tag,
#     tempo,
# ):
    
#     # convert ticks to beats
#     beat_range = [int(tick_range[0] / ec().tokenizer.config["ticks_per_beat"]), int(tick_range[1] / ec().tokenizer.config["ticks_per_beat"])]
#     # remove empty events
#     e = [ev for ev in e if not ev.is_inactive()]

#     # find instruments that are present
#     instruments = set()
#     for i in range(len(e)):
#         instruments = instruments.union(e[i].a["instrument"])
#     # instruments = ec().a["instrument"]

#     # non drum events
#     non_drum_events = [ev for ev in e if "Drums" not in ev.a["instrument"]]

#     # set all tags to tag and all tempos to tempo
#     for i in range(len(e)):
#         e[i].a["tag"] = {tag}
#         e[i].a["tempo"] = {str(ec().quantize_tempo(tempo))}

#     beats = set([str(r) for r in range(beat_range[0], beat_range[1])])
#     pitches = set(
#         [
#             f"{r} (Drums)" if drums else f"{r}"
#             for r in range(pitch_range[0], pitch_range[1])
#         ]
#     )

#     notes_before_removal = len(e)

#     infill_region_pitches = pitch_range[1] - pitch_range[0]

#     # remove if in beat range and pitch range
#     e = [
#         ev
#         for ev in e
#         if not (
#             ev.a["onset/beat"].issubset(beats)
#             and ev.a["pitch"].issubset(pitches)
#         )
#     ]
#     notes_after_removal = len(e)
#     notes_removed = notes_before_removal - notes_after_removal

#     infill_region_ticks = tick_range[1] - tick_range[0]
#     infill_region_bars = (
#         infill_region_ticks / 4 * ec().tokenizer.config["ticks_per_beat"]
#     )
#     # get valid durations
#     valid_durations = set()
#     for d in ec().tokenizer.config["durations"]:
#         if d <= infill_region_bars:
#             valid_durations.add(str(d))

#     print(f"Valid durations: {valid_durations}")

#     valid_onset_beats = {str(r) for r in range(beat_range[0], beat_range[1])}
#     valid_offset_beats = (
#         {"none (Drums)"}
#         if drums
#         else {str(r) for r in range(beat_range[0], beat_range[1] + 1)}
#     )
#     valid_pitches = pitches
#     # valid_durations = {"none (Drums)"} if drums else valid_durations

#     infill_constraint = {
#         "pitch": valid_pitches | {"-"},
#         "onset/beat": valid_onset_beats | {"-"},
#         "offset/beat": valid_offset_beats | {"-"},
#         "instrument": ({"Drums"} if drums else instruments - {"Drums"}) | {"-"},
#         # "duration": valid_durations | {"-"},
#     }

#     # add notes removed with infill constraint
#     e += [ec().intersect(infill_constraint).force_active() for _ in range(20)]

#     # pitch time box size
#     pitch_time_box_notes = int(infill_region_bars * infill_region_pitches / 800)

#     # add 75 optional notes
#     # e += [
#     #     ec().intersect(infill_constraint)
#     #     for _ in range(25)
#     # ]

#     # # pad with empty notes
#     e += [ec().force_inactive() for e in range(n_events - len(e))]

#     # set tag
#     e = [ev.intersect({"tag": {f"{tag}", "-"}}) for ev in e]

#     # set tempo

#     e = [e.intersect(ec().tempo_constraint(tempo)) for e in e]

#     return e
