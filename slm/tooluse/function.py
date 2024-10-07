from core import EventConstraint, tempo_constraint, velocity_constraint, scale_constraint
from core import TOM_PITCHES, HIHAT_PITCHES, PERCUSSION_PITCHES, DRUM_PITCHES, ALL_INSTRUMENTS, ALL_VELOCITIES, ALL_ONSET_TICKS, ALL_OFFSET_TICKS, ALL_ONSET_BEATS, ALL_OFFSET_BEATS, ALL_TEMPOS, ALL_TAGS, ALL_PITCHES, quantize_tempo

N_EVENTS = 300

def simple_beat():
    e = [EventConstraint().force_active() for _ in range(80)]
    # pad
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # tempo to 96 and tag is funk
    e = [ev.intersect(tempo_constraint(148) | {"tag": {"funk", "-"}}) for ev in e]
    return e


def four_on_the_floor_beat():
    e = []
    # add kick on every beat
    for onset_beat in [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"36 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # snares on 2 and 4
    for onset_beat in ["1", "3", "5", "7", "9", "11", "13", "15"]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"38 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # add 40 hihats
    e += [
        EventConstraint().intersect({"pitch": HIHAT_PITCHES | {"-"}}) for _ in range(80)
    ]
    # add percussion
    e += [
        EventConstraint().intersect({"pitch": PERCUSSION_PITCHES | {"-"}})
        for _ in range(20)
    ]
    e += [EventConstraint() for _ in range(N_EVENTS - len(e))]
    # set tempo to 110
    e = [
        ev.intersect(tempo_constraint(130)).intersect({"instrument": {"Drums", "-"}})
        for ev in e
    ]
    return e


# create breakbeat
def breakbeat():
    e = []
    # add 10 kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)"}}).force_active()
        for _ in range(10)
    ]
    # add 10 optional kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)", "-"}}) for _ in range(10)
    ]
    # add 3 toms
    e += [
        EventConstraint().intersect({"pitch": TOM_PITCHES}).force_active()
        for _ in range(10)
    ]

    # add 20 rides
    e += [
        EventConstraint().intersect({"pitch": {"51 (Drums)"}}).force_active()
        for _ in range(40)
    ]
    # 20 optional rides
    e += [
        EventConstraint().intersect({"pitch": {"51 (Drums)", "-"}}) for _ in range(20)
    ]
    # add 10 snare
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)"}}).force_active()
        for _ in range(20)
    ]
    # add 10 optional snares
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)", "-"}}) for _ in range(10)
    ]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # set to 160
    e = [
        ev.intersect(tempo_constraint(95)).intersect({"instrument": {"Drums", "-"}})
        for ev in e
    ]
    e = [ev.intersect({"tag": {"jazz", "-"}}) for ev in e]

    return e


def funk_beat():
    e = []

    # add 10 kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)"}}).force_active()
        for _ in range(20)
    ]

    # add 4 snares
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)"}}).force_active()
        for _ in range(4)
    ]

    # add 10 hihats
    e += [
        EventConstraint().intersect({"pitch": {"42 (Drums)"}}).force_active()
        for _ in range(40)
    ]

    # add 4 open
    e += [
        EventConstraint().intersect({"pitch": {"46 (Drums)"}}).force_active()
        for _ in range(4)
    ]

    # add 10 ghost snare
    e += [
        EventConstraint()
        .intersect({"pitch": {"38 (Drums)"}})
        .intersect(velocity_constraint(40))
        .force_active()
        for _ in range(10)
    ]

    # add up to 20 optional drum notes
    e += [EventConstraint().intersect({"instrument": {"Drums"}}) for _ in range(20)]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    # set to 96
    e = [ev.intersect(tempo_constraint(96)) for ev in e]

    # set tag to funk
    e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]

    return e


def synth_beat():
    e = []
    # add 10 bass
    e += [
        EventConstraint().intersect({"instrument": {"Bass"}}).force_active()
        for _ in range(10)
    ]

    # add 40 synth lead, 4 per beat
    for beat in range(16):
        for tick in range(4):
            e += [
                EventConstraint()
                .intersect({"instrument": {"Synth Lead"}})
                .intersect(
                    {
                        "onset/beat": {str(beat)},
                        "onset/tick": {
                            str(model.tokenizer.config["ticks_per_beat"] // 4 * tick)
                        },
                    }
                )
                .force_active()
            ]

    # add 2 forced synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Synth Lead"}}).force_active()
        for _ in range(2)
    ]

    # add 10 piano
    e += [
        EventConstraint().intersect({"instrument": {"Piano"}}).force_active()
        for _ in range(10)
    ]

    # add 20 drums
    e += [
        EventConstraint().intersect({"instrument": {"Drums"}}).force_active()
        for _ in range(50)
    ]

    # add 50 optional bass or synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Bass", "Drums", "Piano", "-"}})
        for _ in range(100)
    ]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # set to 125
    e = [ev.intersect(tempo_constraint(125)) for ev in e]
    # set tag to pop
    e = [ev.intersect({"tag": {"other", "-"}}) for ev in e]

    return e


# create breakbeat
def prog_beat():
    e = []
    # add 10 kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)"}}).force_active()
        for _ in range(10)
    ]
    # add 10 optional kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)", "-"}}) for _ in range(10)
    ]
    # add 3 toms
    # e += [
    #     EventConstraint().intersect({"pitch": TOM_PITCHES}).force_active()
    #     for _ in range(10)
    # ]

    # add 20 rides
    e += [
        EventConstraint().intersect({"pitch": {"42 (Drums)"}}).force_active()
        for _ in range(20)
    ]
    # 20 optional rides
    e += [
        EventConstraint().intersect({"pitch": {"42 (Drums)", "-"}}) for _ in range(20)
    ]
    # add 10 snare
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)"}}).force_active()
        for _ in range(10)
    ]
    # add 10 optional snares
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)", "-"}}) for _ in range(10)
    ]

    # add 20 bass notes
    e += [
        EventConstraint()
        .intersect({"instrument": {"Bass"}, "pitch": {str(p) for p in range(36, 48)}})
        .force_active()
        for _ in range(30)
    ]
    # add 20 piano notes
    e += [
        EventConstraint().intersect({"instrument": {"Piano"}}).force_active()
        for _ in range(20)
    ]

    # add 20 guitar notes
    e += [
        EventConstraint().intersect({"instrument": {"Guitar"}}).force_active()
        for _ in range(20)
    ]

    e += [
        EventConstraint()
        .intersect(tempo_constraint(160))
        .intersect({"instrument": {"Bass", "Drums", "Piano", "Guitar", "-"}})
        for _ in range(N_EVENTS - len(e))
    ]
    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # set to 160
    # add 50 optional notes

    e = [
        ev.intersect(tempo_constraint(160)).intersect(
            {"instrument": {"Drums", "Bass", "Piano", "Guitar", "-"}}
        )
        for ev in e
    ]

    # constrain to pentatonic scale
    # e = [ev.intersect(
    #     {"pitch":scale_constraint("C pentatonic", (20, 100))["pitch"] | {"-"} | DRUM_PITCHES}
    #     ) for ev in e]
    return e


def four_on_the_floor_beat():
    e = []
    # add kick on every beat
    for onset_beat in [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"36 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # snares on 2 and 4
    for onset_beat in ["1", "3", "5", "7", "9", "11", "13", "15"]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"38 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # add 40 hihats
    e += [
        EventConstraint().intersect({"pitch": HIHAT_PITCHES | {"-"}}) for _ in range(20)
    ]
    # add percussion
    e += [
        EventConstraint().intersect({"pitch": PERCUSSION_PITCHES | {"-"}})
        for _ in range(20)
    ]
    e += [EventConstraint() for _ in range(N_EVENTS - len(e))]
    # set tempo to 110
    e = [
        ev.intersect(tempo_constraint(130)).intersect({"instrument": {"Drums", "-"}})
        for ev in e
    ]
    return e


def disco_beat():
    e = []
    # add kick on every beat
    for onset_beat in [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    ]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"36 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # snares on 2 and 4
    for onset_beat in ["1", "3", "5", "7", "9", "11", "13", "15"]:
        e += [
            EventConstraint()
            .intersect(
                {
                    "pitch": {"38 (Drums)"},
                    "onset/beat": {onset_beat},
                    "onset/tick": {"0"},
                }
            )
            .force_active()
        ]
    # add 40 hihats
    e += [
        EventConstraint().intersect({"pitch": HIHAT_PITCHES | {"-"}}).force_active()
        for _ in range(20)
    ]
    # add percussion
    e += [
        EventConstraint().intersect({"pitch": PERCUSSION_PITCHES | {"-"}})
        for _ in range(20)
    ]

    e += [
        EventConstraint().intersect({"instrument": {"Bass"}}).force_active()
        for _ in range(16)
    ]
    # add 10 piano notes
    e += [
        EventConstraint().intersect({"instrument": {"Piano"}}).force_active()
        for _ in range(20)
    ]
    # add 10 guitar notes
    e += [
        EventConstraint().intersect({"instrument": {"Guitar"}}).force_active()
        for _ in range(10)
    ]
    # add 10 synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Synth Lead"}}).force_active()
        for _ in range(10)
    ]
    # add 50 blank notes
    e += [EventConstraint().force_inactive() for _ in range(50)]
    e += [
        EventConstraint().intersect(
            {"instrument": {"Bass", "Drums", "Piano", "Guitar", "Synth Lead", "-"}}
        )
        for _ in range(N_EVENTS - len(e))
    ]
    # set tempo to 110
    # add 30 bass notes

    e = [
        ev.intersect(tempo_constraint(130)).intersect(
            {"instrument": {"Bass", "Drums", "Piano", "Guitar", "Synth Lead", "-"}}
        )
        for ev in e
    ]
    # add pop tag
    e = [ev.intersect({"tag": {"pop", "-"}}) for ev in e]
    return e


def metal_beat():
    e = []

    # add 10 kicks
    e += [
        EventConstraint().intersect({"pitch": {"36 (Drums)"}}).force_active()
        for _ in range(12)
    ]

    # add 4 snares
    e += [
        EventConstraint().intersect({"pitch": {"38 (Drums)"}}).force_active()
        for _ in range(4)
    ]

    # add 10 hihats
    e += [
        EventConstraint().intersect({"pitch": HIHAT_PITCHES}).force_active()
        for _ in range(10)
    ]

    # add up to 20 optional drum notes
    e += [EventConstraint().intersect({"instrument": {"Drums"}}) for _ in range(10)]

    # add 30 guitar notes
    e += [
        EventConstraint()
        .intersect({"instrument": {"Guitar"}})
        # .intersect(scale_constraint("E pentatonic", (30, 100)))
        .force_active()
        for _ in range(50)
    ]

    # add 20 bass notes
    e += [
        EventConstraint()
        .intersect({"instrument": {"Bass"}})
        # .intersect(scale_constraint("E pentatonic", (30, 100)))
        .force_active()
        for _ in range(20)
    ]

    # add 30 optional guitar or bass notes
    e += [
        EventConstraint().intersect({"instrument": {"Guitar", "Bass", "-"}})
        # .intersect(scale_constraint("E pentatonic", (30, 100)))
        for _ in range(30)
    ]

    # add 40 optional notes, guitar drums or bass
    e += [
        EventConstraint().intersect({"instrument": {"Guitar", "Drums", "Bass", "-"}})
        for _ in range(80)
    ]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # set tempo to 160
    e = [ev.intersect(tempo_constraint(150)) for ev in e]
    # set tag to metal
    # e = [ev.intersect({"tag": {"metal", "-"}}) for ev in e]

    return e


def fun_beat():
    e = []
    # add 30 piano
    e += [
        EventConstraint()
        .intersect({"instrument": {"Chromatic Percussion"}})
        .force_active()
        for _ in range(20)
    ]
    # add 10 low velocity Chromatic Percussion
    e += [
        EventConstraint()
        .intersect({"instrument": {"Chromatic Percussion"}})
        .intersect(velocity_constraint(100))
        .intersect({"pitch": {str(s) for s in range(50, 100)}})
        .force_active()
        for _ in range(10)
    ]
    # add 10 high velocity Chromatic Percussion
    e += [
        EventConstraint()
        .intersect({"instrument": {"Chromatic Percussion"}})
        .intersect(velocity_constraint(40))
        .force_active()
        for _ in range(10)
    ]

    # add 30 guitar
    # e += [
    #     EventConstraint().intersect({"instrument": {"Guitar"}}).force_active()
    #     for _ in range(30)
    # ]

    # add 5 bass
    # e += [
    #     EventConstraint().intersect({"instrument": {"Bass"}}).force_active()
    #     for _ in range(5)
    # ]

    # add 10 synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Synth Pad"}}).force_active()
        for _ in range(16)
    ]

    # add 50 optional
    e += [
        EventConstraint()
        .intersect({"instrument": {"Bass", "-"}})
        .intersect(scale_constraint("C pentatonic", (30, 50)))
        for _ in range(20)
    ]
    # add one bass note on first beat
    e += [
        EventConstraint()
        .intersect({"instrument": {"Bass"}})
        .intersect(scale_constraint("C pentatonic", (30, 50)))
        .force_active()
        for _ in range(10)
    ]

    # add 40 drums

    # constrain to major pitch set
    e = [ev.intersect(scale_constraint("C major", (20, 100))) for ev in e]
    # pad with empty notes
    e += [
        EventConstraint()
        .intersect({"instrument": {"Drums"}, "pitch": PERCUSSION_PITCHES})
        .force_active()
        for _ in range(40)
    ]
    # add 30 drum notes
    e += [
        EventConstraint()
        .intersect(
            {"instrument": {"Drums"}, "pitch": DRUM_PITCHES - PERCUSSION_PITCHES}
        )
        .force_active()
        for _ in range(30)
    ]

    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    # set tag to pop
    e = [
        ev.intersect({"tag": {"other", "-"}}).intersect(tempo_constraint(120))
        for ev in e
    ]
    return e


def simple_beat():
    e = []
    # add 30 piano
    e += [
        EventConstraint().intersect({"instrument": {"Piano"}}).force_active()
        for _ in range(20)
    ]
    # add 10 low velocity piano
    e += [
        EventConstraint()
        .intersect({"instrument": {"Piano"}})
        .intersect(velocity_constraint(100))
        .intersect({"pitch": {str(s) for s in range(50, 100)}})
        .force_active()
        for _ in range(10)
    ]
    # add 10 high velocity piano
    e += [
        EventConstraint()
        .intersect({"instrument": {"Piano"}})
        .intersect(velocity_constraint(40))
        .force_active()
        for _ in range(10)
    ]

    # add 30 guitar
    # e += [
    #     EventConstraint().intersect({"instrument": {"Guitar"}}).force_active()
    #     for _ in range(30)
    # ]

    # add 5 bass
    # e += [
    #     EventConstraint().intersect({"instrument": {"Bass"}}).force_active()
    #     for _ in range(5)
    # ]

    # add 10 synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Synth Lead"}}).force_active()
        for _ in range(10)
    ]

    # add 50 optional
    e += [
        EventConstraint().intersect({"instrument": {"Piano", "-"}}) for _ in range(20)
    ]

    # constrain to major pitch set
    e = [ev.intersect(scale_constraint("C major", (20, 100))) for ev in e]
    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    # set tag to pop
    e = [
        ev.intersect({"tag": {"pop", "-"}}).intersect(tempo_constraint(90)) for ev in e
    ]

    return e


def infill(e, beat_range, pitch_range, drums, tag="other", tempo=120):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(quantize_tempo(tempo))}

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
    # e += [EventConstraint().force_inactive() for _ in range(40)]

    infill_constraint = {
        "pitch": {
            f"{str(r)}{' (Drums)' if drums else ''}"
            for r in range(pitch_range[0], pitch_range[1])
        }
        | {"-"},
        "onset/beat": {str(r) for r in range(beat_range[0], beat_range[1])} | {"-"},
        "offset/beat": {str(r) for r in range(beat_range[0], beat_range[1])} | {"-"},
        "instrument": ({"Drums"} if drums else ALL_INSTRUMENTS - {"Drums"}) | {"-"},
        "tag": {tag, "-"},
        "tempo": {str(quantize_tempo(tempo)), "-"},
    }

    # count notes per beat

    # add between notes_to_remove - 10 and notes_to_remove + 10 notes. At least 10 notes
    # lower_bound_notes = max(notes_removed - 10, 10)
    # upper_bound_notes = notes_removed + 10
    # add between 0 and
    # add 3 forced active
    e += [
        EventConstraint().intersect(infill_constraint).force_active() for _ in range(3)
    ]
    if notes_removed > 0:
        e += [
            EventConstraint().intersect(infill_constraint).force_active()
            for _ in range(notes_removed // 2)
        ]
        e += [
            EventConstraint().intersect(infill_constraint) for _ in range(notes_removed)
        ]
    #

    print(f"Notes removed: {notes_removed}")

    # # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    # add 10 empty notes
    # e += [EventConstraint().force_inactive() for _ in range(40)]

    return e


def repitch(e, beat_range, pitch_range, drums, tag="other", tempo=120):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(quantize_tempo(tempo))}

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
    e += [EventConstraint().force_inactive() for e in range(N_EVENTS - len(e))]
    return e


def revelocity(e, beat_range, pitch_range, drums, tag="other", tempo=120):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(quantize_tempo(tempo))}

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
            e[i].a["velocity"] = ALL_VELOCITIES
    # pad with empty notes
    e += [EventConstraint().force_inactive() for e in range(N_EVENTS - len(e))]
    return e


def retime(e, beat_range, pitch_range, drums, tag="other", tempo=120):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(quantize_tempo(tempo))}

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
            e[i].a["onset/tick"] = ALL_ONSET_TICKS
            e[i].a["offset/tick"] = ALL_OFFSET_TICKS

    # pad with empty notes
    e += [EventConstraint().force_inactive() for e in range(N_EVENTS - len(e))]
    return e


def reinstrument(e, beat_range, pitch_range, drums, tag="other", tempo=120):
    # remove empty events
    e = [ev for ev in e if not ev.is_inactive()]

    # set all tags to tag and all tempos to tempo
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
        e[i].a["tempo"] = {str(quantize_tempo(tempo))}

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
            e[i].a["instrument"] = {"Drums"} if drums else ALL_INSTRUMENTS - {"Drums"}

    # pad with empty notes
    e += [EventConstraint().force_inactive() for e in range(N_EVENTS - len(e))]
    return e


def add_snare_ghost_notes(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    e += [
        EventConstraint()
        .intersect({"pitch": {"38 (Drums)", "-"}} | velocity_constraint(50))
        .force_active()
        for _ in range(5)
    ]

    e += [
        EventConstraint().intersect(
            {"pitch": {"38 (Drums)", "-"}} | velocity_constraint(80)
        )
        for _ in range(20)
    ]
    # pad
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    return e


def add_percussion(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # remove percussion
    e = [ev for ev in e if ev.a["pitch"].isdisjoint(PERCUSSION_PITCHES)]

    e += [
        EventConstraint()
        .intersect({"pitch": PERCUSSION_PITCHES | {"-"}})
        .force_active()
        for _ in range(5)
    ]

    e += [
        EventConstraint().intersect({"pitch": PERCUSSION_PITCHES | {"-"}})
        for _ in range(20)
    ]
    # pad
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    return e


def add_tom_fill(e):
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
        EventConstraint()
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
        EventConstraint().intersect(
            {"instrument": {"Drums"}, "onset/beat": {"14", "15", "_"}}
        )
        for _ in range(10)
    ]
    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e


def add_dynamic_hihats(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]
    # remove hihats
    e = [ev for ev in e if ev.a["pitch"].isdisjoint(HIHAT_PITCHES)]
    e += [
        EventConstraint()
        .intersect({"pitch": HIHAT_PITCHES} | velocity_constraint(30))
        .force_active()
        for _ in range(3)
    ]

    e += [
        EventConstraint()
        .intersect({"pitch": HIHAT_PITCHES} | velocity_constraint(60))
        .force_active()
        for _ in range(3)
    ]

    # add up to 10 more
    e += [
        EventConstraint().intersect({"pitch": HIHAT_PITCHES | {"-"}})
        for _ in range(N_EVENTS - len(e))
    ]

    # pad
    # e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    return e


def add_arpeggio(e):
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]

    # remove all sytnh lead
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Synth Lead"})]

    # add synth
    # add 40 synth lead, 4 per beat
    for beat in range(16):
        for tick in range(4):
            e += [
                EventConstraint()
                .intersect({"instrument": {"Synth Lead"}})
                .intersect(
                    {
                        "onset/beat": {str(beat)},
                        "onset/tick": {
                            str(model.tokenizer.config["ticks_per_beat"] // 4 * tick)
                        },
                        "pitch": {str(note) for note in range(50, 100)},
                    }
                )
                .force_active()
            ]

    # add 2 forced synth lead
    e += [
        EventConstraint().intersect({"instrument": {"Synth Lead"}}).force_active()
        for _ in range(2)
    ]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e


def add_lead(e):
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
        EventConstraint()
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
        EventConstraint().intersect(
            {
                "instrument": {instrument, "-"},
                "pitch": {str(note) for note in range(40, 100)} | {"-"},
            }
        )
        for _ in range(20)
    ]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    # add tag constraint
    e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

    return e


# add some chords
def add_chords(e):
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
        EventConstraint()
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
    e += [
        EventConstraint().intersect({"instrument": {instrument, ""}}) for _ in range(30)
    ]

    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]

    # add tag constraint
    e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

    return e


# add bassline that matches kick
def add_locked_in_bassline(e):
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
            EventConstraint()
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
    e += [EventConstraint().intersect({"instrument": {"Bass"}}) for _ in range(5)]
    # pad with empty notes
    e += [EventConstraint().force_inactive() for _ in range(N_EVENTS - len(e))]
    return e
