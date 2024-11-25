from .core import HIHAT_PITCHES, DRUM_PITCHES, PERCUSSION_PITCHES, TOM_PITCHES


def funky_bassline(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # remove all bass
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]

    # # add 10 bass notes under 36
    # e += [
    #     ec()
    #     .intersect({"instrument": {"Bass"}})
    #     .force_active()
    #     for _ in range(10)
    # ]

    # add 10 bass notes over 36
    e += [
        ec()
        .intersect({"instrument": {"Bass", "-"}, 
                    "pitch": {str(p) for p in range(23, 65)} | {"-"},
        })
        .force_active()
        for _ in range(30)
    ]


    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    return e


def add_snare_ghost_notes(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    e += [
        ec()
        .intersect({"pitch": {"38 (Drums)", "-"}} | ec().velocity_constraint(50))
        .force_active()
        for _ in range(5)
    ]

    e += [
        ec().intersect({"pitch": {"38 (Drums)", "-"}} | ec().velocity_constraint(80))
        for _ in range(20)
    ]
    # pad
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    return e


def add_percussion(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # remove percussion
    e = [ev for ev in e if ev.a["pitch"].isdisjoint(PERCUSSION_PITCHES)]

    e += [
        ec().intersect({"pitch": PERCUSSION_PITCHES | {"-"}}).force_active()
        for _ in range(5)
    ]

    e += [ec().intersect({"pitch": PERCUSSION_PITCHES | {"-"}}) for _ in range(20)]
    # pad
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    return e


def add_tom_fill(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]
    # remove drums in last 4 bars
    e = [
        ev
        for ev in e
        if not (
            not ev.a["onset/beat"].isdisjoint({"14", "15"})
            and not ev.a["instrument"].isdisjoint({"Drums"})
        )
    ]
    # add 3 toms from any of the tom pitches.
    e += [
        ec()
        .intersect(
            {
                "instrument": {"Drums"},
                "pitch": TOM_PITCHES,
                "onset/beat": {"14", "15", "_"},
            }
        )
        .force_active()
        for e in range(3)
    ]
    # add up to 10 more drums
    e += [
        ec().intersect({"instrument": {"Drums"}, "onset/beat": {"14", "15", "_"}})
        for _ in range(10)
    ]
    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e


def add_dynamic_hihats(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]
    # remove hihats
    e = [ev for ev in e if ev.a["pitch"].isdisjoint(HIHAT_PITCHES)]
    e += [
        ec()
        .intersect({"pitch": HIHAT_PITCHES} | ec().velocity_constraint(30))
        .force_active()
        for _ in range(3)
    ]

    e += [
        ec()
        .intersect({"pitch": HIHAT_PITCHES} | ec().velocity_constraint(60))
        .force_active()
        for _ in range(3)
    ]

    # add up to 10 more
    e += [
        ec().intersect({"pitch": HIHAT_PITCHES | {"-"}})
        for _ in range(n_events - len(e))
    ]

    # pad
    # e += [ec().force_inactive() for _ in range(n_events - len(e))]

    return e


def add_arpeggio(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # remove all sytnh lead
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Synth Lead"})]

    # add synth
    # add 40 synth lead, 4 per beat
    for beat in range(16):
        for tick in range(4):
            e += [
                ec()
                .intersect({"instrument": {"Synth Lead"}})
                .intersect(
                    {
                        "onset/beat": {str(beat)},
                        "onset/tick": {
                            str(ec().tokenizer.config["ticks_per_beat"] // 4 * tick)
                        },
                        "pitch": {str(note) for note in range(50, 100)},
                    }
                )
                .force_active()
            ]

    # add 2 forced synth lead
    e += [
        ec().intersect({"instrument": {"Synth Lead"}}).force_active() for _ in range(2)
    ]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e


def add_lead(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    tag = "pop"
    instrument = "Synth Lead"

    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # set all tags to jazz
    for i in range(len(e)):
        e[i].a["tag"] = {tag}

    # remove piano
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({instrument})]

    e += [
        ec()
        .intersect(
            {
                "instrument": {instrument},
                "pitch": {str(note) for note in range(55, 100)},
            }
        )
        .force_active()
    ]

    # add optional Brass notes
    e += [
        ec().intersect(
            {
                "instrument": {instrument, "-"},
                "pitch": {str(note) for note in range(40, 100)} | {"-"},
            }
        )
        for _ in range(20)
    ]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    # add tag constraint
    e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

    return e


# add some chords
def add_chords(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    tag = "pop"

    instrument = "Piano"

    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # set all tags to jazz
    for i in range(len(e)):
        e[i].a["tag"] = {tag}

    # remove piano
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({instrument})]

    # add a 3 Guitar notes on first beat
    e += [
        ec()
        .intersect(
            {
                "instrument": {instrument},
                "onset/beat": {"0"},
                "offset/beat": {"2", "3", "4"},
            }  #   | scale_constraint("C major", (50,100))
        )
        .force_active()
        for i in range(3)
    ]

    # add optionalinstrumentnotes
    e += [ec().intersect({"instrument": {instrument, ""}}) for _ in range(30)]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    # add tag constraint
    e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

    return e


# add bassline that matches kick
def add_locked_in_bassline(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
    # remove inactive notes
    e = [ev for ev in e if ev.is_active()]
    # remove bass
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
    # find kicks
    kicks = [
        ev
        for ev in e
        if {"35 (Drums)", "36 (Drums)", "37 (Drums)"}.intersection(ev.a["pitch"])
    ]
    # add bass on every kick
    for kick in kicks:
        e += [
            ec()
            .intersect(
                {
                    "instrument": {"Bass"},
                    "onset/beat": kick.a["onset/beat"],
                    "onset/tick": kick.a["onset/tick"],
                }
            )
            .force_active()
        ]
    # add up to 5 more bass notes
    e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(5)]
    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e
