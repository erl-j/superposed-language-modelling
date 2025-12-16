from .core import HIHAT_PITCHES, DRUM_PITCHES, TOM_PITCHES, PERCUSSION_PITCHES, CRASH_PITCHES

def unconditional(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
    e = [ec() for _ in range(n_events)]
    return e

def pentatonic_guitar(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
    e = []
    # add forced guitar notes with pentatonic scale constraint
    e += [ec().intersect({"instrument": {"Guitar"}})
    .intersect(ec().pitch_in_scale_constraint("C major pentatonic", (20, 100)))
    .force_active() for _ in range(20)]
    # add optional guitar notes with pentatonic scale constraint
    e += [ec().intersect({"instrument": {"Guitar", "-"}})
    .intersect(ec().pitch_in_scale_constraint("C major pentatonic", (20, 100)))
    .force_active() for _ in range(60)]
    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

def extreme_drums(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
    '''
    # this constraint create an extreme drum beat with a lot of notes. 
    # many kicks first of all
    # some snares
    # a lot of toms as well
    # some cymbals as well
    # set tag to metal
    '''
    e = []
    # add many kicks
    e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(30)]
    # add some snares
    e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(8)]
    # add many toms
    e += [ec().intersect({"pitch": TOM_PITCHES}).force_active() for _ in range(30)]
    # add some rides
    e += [ec().intersect({"pitch": CRASH_PITCHES}).force_active() for _ in range(10)]

    # add 20 drums
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for _ in range(32)]
    # add 50 optional drums
    e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(32)]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    # set tempo to 120
    e = [ev.intersect(ec().tempo_constraint(75)) for ev in e]
    # add tag constraint
    e = [ev.intersect({"tag": {"metal", "-"}}) for ev in e]
    return e

def ballad(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo,
):
    # remove current notes
    e = []
    # add 20 piano notes
    e += [ec().intersect({"instrument": {"Piano"}}).force_active() for _ in range(64)]
    # add 20 bass notes
    e += [ec().intersect({"instrument": {"Bass"}}).force_active() for _ in range(16)]
    # add 100 optional notes
    e += [ec().intersect({"instrument": {"Piano","Bass","-"}}) for _ in range(64)]
    # add pitch constraint to all notes
    e = [ev.intersect(ec().pitch_in_scale_constraint("C major", (20, 100))) for ev in e]
    # add 20 drums
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for _ in range(32)]
    # add 50 optional drums
    e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(32)]
    # set tempo to 120
    e = [ev.intersect(ec().tempo_constraint(75)) for ev in e]
    # add tag constraint
    e = [ev.intersect({"tag": {"pop", "-"}}) for ev in e]
    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e


def bass_and_drums(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
    e = []
    # force 1 bass note
    e += [ec().intersect({"instrument": {"Bass"}}).force_active() for _ in range(3)]
    # force 1 drums note
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for _ in range(3)]

    # constrain instrument to be only bass and drums
    e += [ec().intersect({"instrument": {"Bass", "Drums"}}).force_active() for i in range(50)]

    # add 50 optional bass and drums
    e += [ec().intersect({"instrument": {"Bass", "Drums", "-"}}) for i in range(50)]
    # pad
    # set tag to pop
    e = [ev.intersect({"tag": {"rock", "-"}}) for ev in e]
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e

def strings_and_flute(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
    e = []

    e += [ec().intersect({"instrument": {"Flute", "Strings"}}) for _ in range(n_events)]

    e = [ev.intersect({"tag": {"classical", "-"}}) for ev in e]
    return e


def ballad(e, ec, n_events):
    # remove current notes
    e = []
    # add 64 piano notes
    e += [ec().intersect({"instrument": {"Piano"}}).force_active() for _ in range(64)]
    # add 16 bass notes
    e += [ec().intersect({"instrument": {"Bass"}}).force_active() for _ in range(16)]
    # add 64 optional notes
    e += [ec().intersect({"instrument": {"Piano","Bass","-"}}) for _ in range(64)]
    # add pitch constraint to all notes
    e = [ev.intersect(ec().pitch_in_scale_constraint("C major", (20, 100))) for ev in e]
    # add 32 drums
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for _ in range(32)]
    # add 32 optional drums
    e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(32)]
    # set tempo to 75
    e = [ev.intersect(ec().tempo_constraint(75)) for ev in e]
    # add tag constraint
    e = [ev.intersect({"tag": {"pop", "-"}}) for ev in e]
    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e


def jazz_piano(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
) : 
    # remove empty notes
    e = [ev for ev in e if ev.is_active()]
    # remove all piano
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
    # add between 50 piano notes
    e += [ec().intersect({"instrument": {"Piano"}}).force_active() for _ in range(50)]
    # add 50 optional piano notes
    e += [ec().intersect({"instrument": {"Piano", "-"}}) for _ in range(50)]
    # set tempo to 120
    e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
    # add tag constraint
    e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]
    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e




def reggaeton_beat(
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
    tag = "jazz"
    # set tag to jazz
    for i in range(len(e)):
        e[i].a["tag"] = {tag}
    # remove drums
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
    # add 10 kicks
    e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(20)]
    # add 10 optional kicks
    e += [ec().intersect({"pitch": {"36 (Drums)", "-"}}) for _ in range(10)]
    # add 3 toms
    e += [ec().intersect({"pitch": TOM_PITCHES}).force_active() for _ in range(10)]

    # add ride 
    RIDE_PITCHES = {"51 (Drums)", "53 (Drums)", "59 (Drums)"}
    # add 20 rides
    e += [ec().intersect({"pitch":RIDE_PITCHES }).force_active() for _ in range(40)]
    # 20 optional rides
    e += [ec().intersect({"pitch": RIDE_PITCHES | {"-"}}) for _ in range(20)]
    # add 10 snare
    e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(6)]
    # add 10 optional snares
    e += [ec().intersect({"pitch": {"38 (Drums)", "-"}}) for _ in range(10)]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    # set to 160
    # e = [
    #     ev.intersect(ec().tempo_constraint(95)).intersect(
    #         {"instrument": {"Drums", "-"}}
    #     )
    #     for ev in e
    # ]
    e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

    # set tempo
    e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

    # tempo


    return e
    # e = []

    # # remove all drums
    # e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]

    # # Dembow rhythm pattern (repeated for each bar)
    # # first, kick on every beat of every bar
    # for bar in range(4):
    #     for beat in range(4):
    #         e += [
    #             ec()
    #             .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat + (bar * 4))},"onset/tick": {"0"}})
    #             .force_active()
    #         ]

    #     e += [
    #         ec()
    #         .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str( (bar * 4))}, "onset/tick": {"18"}})
    #     ]
    #     e += [
    #         ec()
    #         .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(1 +  (bar * 4))}, "onset/tick": {"12"}})
    #     ]
    #     e += [
    #         ec()
    #         .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(2 +  (bar * 4))}, "onset/tick": {"18"}})
    #     ]
    #     e += [
    #         ec()
    #         .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(3 +  (bar * 4))}, "onset/tick": {"12"}})
    #     ]

    # # add 10 active hihats
    # e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(10)]

    # # # add 60 optional drums that are not kicks or snare
    # e += [ec().intersect({"instrument": {"Drums"}}).intersect({"pitch": DRUM_PITCHES - {"36 (Drums)", "38 (Drums)"}}) for _ in range(30)]

    # # set tempo to 110
    # e = [ev.intersect(ec().tempo_constraint(90)) for ev in e]

    # # set tag to latin
    # e = [ev.intersect({"tag": {"latin", "-"}}) for ev in e]

    # # pad with empty notes
    # e += [ec().force_inactive() for _ in range(n_events - len(e))]

    # return e

def funk_beat(
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

    # remove all drums
    e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]

    # add 10 kicks
    e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(20)]

    # add 4 snares
    e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(4)]

    # add 10 hihats
    e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(40)]

    # add 4 open
    e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]

    # add 10 ghost snare
    e += [
        ec()
        .intersect({"pitch": {"38 (Drums)"}})
        .intersect(ec().velocity_constraint(40))
        .force_active()
        for _ in range(10)
    ]

    # add up to 20 optional drum notes
    e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    # set to 96
    e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

    # set tag to funk
    e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

    return e

# create breakbeat
def breakbeat(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    e = []
    # add 10 kicks
    e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(20)]
    # add 10 optional kicks
    e += [ec().intersect({"pitch": {"36 (Drums)", "-"}}) for _ in range(10)]
    # add 3 toms
    e += [ec().intersect({"pitch": TOM_PITCHES}).force_active() for _ in range(10)]

    # add 20 rides
    e += [ec().intersect({"pitch": HIHAT_PITCHES}).force_active() for _ in range(40)]
    # 20 optional rides
    e += [ec().intersect({"pitch": HIHAT_PITCHES |{"-"}}) for _ in range(20)]
    # add 10 snare
    e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(8)]
    # add 10 optional snares
    e += [ec().intersect({"pitch": {"38 (Drums)", "-"}}) for _ in range(10)]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    # set to 160
    e = [
        ev.intersect(ec().tempo_constraint(95)).intersect(
            {"instrument": {"Drums", "-"}}
        )
        for ev in e
    ]
    e = [ev.intersect({"tag": {"metal", "-"}}) for ev in e]

    return e


def triplet_drums(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # one drum note every 8 ticks (triplet feel)
    ticks_per_beat = ec().tokenizer.config["ticks_per_beat"]

    n_beats = 16
    triplet_ticks = {str(t) for t in range(0, ticks_per_beat * n_beats, ticks_per_beat // 3)}

    e = [
        ec().intersect({"instrument": {"Drums"}, "onset/global_tick": {str(onset_tick)}}).force_active() 
        for onset_tick in range(0, ticks_per_beat * n_beats, ticks_per_beat // 3)
    ]

    # add 50 more drums on any triplet
    e += [ec().intersect({"instrument": {"Drums"}, "onset/global_tick": triplet_ticks}).force_active() for _ in range(50)]
    
    # set tempo
    e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
    # set tag
    e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]
    # pad
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    return e


# get all possible values for each attribute
def simple_beat(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    e = [ec().force_active() for _ in range(80)]
    # pad
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    # tempo to 96 and tag is funk
    e = [ev.intersect(ec().tempo_constraint(148) | {"tag": {"funk", "-"}}) for ev in e]
    return e


# create breakbeat
def prog_beat(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    e = []
    # add 10 kicks
    e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(10)]
    # add 10 optional kicks
    e += [ec().intersect({"pitch": {"36 (Drums)", "-"}}) for _ in range(10)]
    # add 3 toms
    # e += [
    #     ec().intersect({"pitch": TOM_PITCHES}).force_active()
    #     for _ in range(10)
    # ]

    # add 20 rides
    e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(20)]
    # 20 optional rides
    e += [ec().intersect({"pitch": {"42 (Drums)", "-"}}) for _ in range(20)]
    # add 10 snare
    e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(10)]
    # add 10 optional snares
    e += [ec().intersect({"pitch": {"38 (Drums)", "-"}}) for _ in range(10)]

    # add 20 bass notes
    e += [
        ec()
        .intersect({"instrument": {"Bass"}, "pitch": {str(p) for p in range(36, 48)}})
        .force_active()
        for _ in range(30)
    ]
    # add 20 piano notes
    e += [ec().intersect({"instrument": {"Piano"}}).force_active() for _ in range(20)]

    # add 20 guitar notes
    e += [ec().intersect({"instrument": {"Guitar"}}).force_active() for _ in range(20)]

    e += [
        ec()
        .intersect(ec().tempo_constraint(160))
        .intersect({"instrument": {"Bass", "Drums", "Piano", "Guitar", "-"}})
        for _ in range(n_events - len(e))
    ]
    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    # set to 160
    # add 50 optional notes

    e = [
        ev.intersect(ec().tempo_constraint(160)).intersect(
            {"instrument": {"Drums", "Bass", "Piano", "Guitar", "-"}}
        )
        for ev in e
    ]

    # constrain to pentatonic scale
    # e = [ev.intersect(
    #     {"pitch":scale_constraint("C pentatonic", (20, 100))["pitch"] | {"-"} | DRUM_PITCHES}
    #     ) for ev in e]
    return e


def four_on_the_floor_beat(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
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
            ec()
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
            ec()
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
    e += [ec().intersect({"pitch": HIHAT_PITCHES | {"-"}}) for _ in range(20)]
    # add percussion
    e += [ec().intersect({"pitch": PERCUSSION_PITCHES | {"-"}}) for _ in range(20)]
    e += [ec() for _ in range(n_events - len(e))]
    # set tempo to 110
    e = [
        ev.intersect(ec().tempo_constraint(130)).intersect(
            {"instrument": {"Drums", "-"}}
        )
        for ev in e
    ]
    return e


def disco_beat(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
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
            ec()
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
            ec()
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
        ec().intersect({"pitch": HIHAT_PITCHES | {"-"}}).force_active()
        for _ in range(20)
    ]
    # add percussion
    e += [ec().intersect({"pitch": PERCUSSION_PITCHES | {"-"}}) for _ in range(20)]

    e += [ec().intersect({"instrument": {"Bass"}}).force_active() for _ in range(16)]
    # add 10 piano notes
    e += [ec().intersect({"instrument": {"Piano"}}).force_active() for _ in range(20)]
    # add 10 guitar notes
    e += [ec().intersect({"instrument": {"Guitar"}}).force_active() for _ in range(10)]
    # add 10 synth lead
    e += [
        ec().intersect({"instrument": {"Synth Lead"}}).force_active() for _ in range(10)
    ]
    # add 50 blank notes
    e += [ec().force_inactive() for _ in range(50)]
    e += [
        ec().intersect(
            {"instrument": {"Bass", "Drums", "Piano", "Guitar", "Synth Lead", "-"}}
        )
        for _ in range(n_events - len(e))
    ]
    # set tempo to 110
    # add 30 bass notes

    e = [
        ev.intersect(ec().tempo_constraint(130)).intersect(
            {"instrument": {"Bass", "Drums", "Piano", "Guitar", "Synth Lead", "-"}}
        )
        for ev in e
    ]
    # add pop tag
    e = [ev.intersect({"tag": {"pop", "-"}}) for ev in e]
    return e


def metal_beat(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    e = []

    # add 10 kicks
    e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(12)]

    # add 4 snares
    e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(4)]

    # add 10 hihats
    e += [ec().intersect({"pitch": HIHAT_PITCHES}).force_active() for _ in range(10)]

    # add up to 20 optional drum notes
    e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]

    # add 30 guitar notes
    e += [
        ec()
        .intersect({"instrument": {"Guitar"}})
        # .intersect(scale_constraint("E pentatonic", (30, 100)))
        .force_active()
        for _ in range(50)
    ]

    # add 20 bass notes
    e += [
        ec()
        .intersect({"instrument": {"Bass"}})
        # .intersect(scale_constraint("E pentatonic", (30, 100)))
        .force_active()
        for _ in range(20)
    ]

    # add 30 optional guitar or bass notes
    e += [
        ec().intersect({"instrument": {"Guitar", "Bass", "-"}})
        # .intersect(scale_constraint("E pentatonic", (30, 100)))
        for _ in range(30)
    ]

    # add 40 optional notes, guitar drums or bass
    e += [
        ec().intersect({"instrument": {"Guitar", "Drums", "Bass", "-"}})
        for _ in range(80)
    ]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    # set tempo to 160
    e = [ev.intersect(ec().tempo_constraint(150)) for ev in e]
    # set tag to metal
    # e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

    return e


def fun_beat(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    e = []
    # add 30 piano
    e += [
        ec().intersect({"instrument": {"Chromatic Percussion"}}).force_active()
        for _ in range(20)
    ]
    # add 10 low velocity Chromatic Percussion
    e += [
        ec()
        .intersect({"instrument": {"Chromatic Percussion"}})
        .intersect(ec().velocity_constraint(100))
        .intersect({"pitch": {str(s) for s in range(50, 100)}})
        .force_active()
        for _ in range(10)
    ]
    # add 10 high velocity Chromatic Percussion
    e += [
        ec()
        .intersect({"instrument": {"Chromatic Percussion"}})
        .intersect(ec().velocity_constraint(40))
        .force_active()
        for _ in range(10)
    ]

    # add 30 guitar
    # e += [
    #     ec().intersect({"instrument": {"Guitar"}}).force_active()
    #     for _ in range(30)
    # ]

    # add 5 bass
    # e += [
    #     ec().intersect({"instrument": {"Bass"}}).force_active()
    #     for _ in range(5)
    # ]

    # add 10 synth lead
    e += [
        ec().intersect({"instrument": {"Synth Pad"}}).force_active() for _ in range(16)
    ]

    # add 50 optional
    e += [
        ec()
        .intersect({"instrument": {"Bass", "-"}})
        .intersect(ec().pitch_in_scale_constraint("C pentatonic", (30, 50)))
        for _ in range(20)
    ]
    # add one bass note on first beat
    e += [
        ec()
        .intersect({"instrument": {"Bass"}})
        .intersect(ec().pitch_in_scale_constraint("C pentatonic", (30, 50)))
        .force_active()
        for _ in range(10)
    ]

    # add 40 drums

    # constrain to major pitch set
    e = [ev.intersect(ec().pitch_in_scale_constraint("C major", (20, 100))) for ev in e]
    # pad with empty notes
    e += [
        ec()
        .intersect({"instrument": {"Drums"}, "pitch": PERCUSSION_PITCHES})
        .force_active()
        for _ in range(40)
    ]
    # add 30 drum notes
    e += [
        ec()
        .intersect(
            {"instrument": {"Drums"}, "pitch": DRUM_PITCHES - PERCUSSION_PITCHES}
        )
        .force_active()
        for _ in range(30)
    ]

    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    # set tag to pop
    e = [
        ev.intersect({"tag": {"other", "-"}}).intersect(ec().tempo_constraint(120))
        for ev in e
    ]
    return e


def synth_beat(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    e = []
    # add 10 bass
    e += [ec().intersect({"instrument": {"Bass"}}).force_active() for _ in range(10)]

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
                    }
                )
                .force_active()
            ]

    # add 2 forced synth lead
    e += [
        ec().intersect({"instrument": {"Synth Lead"}}).force_active() for _ in range(2)
    ]

    # add 10 piano
    e += [ec().intersect({"instrument": {"Piano"}}).force_active() for _ in range(10)]

    # add 20 drums
    e += [ec().intersect({"instrument": {"Drums"}}).force_active() for _ in range(50)]

    # add 50 optional bass or synth lead
    e += [
        ec().intersect({"instrument": {"Bass", "Drums", "Piano", "-"}})
        for _ in range(100)
    ]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]
    # set to 125
    e = [ev.intersect(ec().tempo_constraint(125)) for ev in e]
    # set tag to pop
    e = [ev.intersect({"tag": {"other", "-"}}) for ev in e]

    return e

def band_beat(
    e,
    ec,
    n_events,
    beat_range,
    pitch_range,
    drums,
    tag,
    tempo,
):
    # add
    e = []
    # add 20 bass
    e += [ec().intersect({"instrument": {"Bass"}}).force_active()
           for _ in range(20)]
    # add 40 piano
    e += [ec().intersect({"instrument": {"Piano"}}).force_active()
           for _ in range(40)]

    # add 70 drums
    e += [ec().intersect({"instrument": {"Drums"}}).force_active()
           for _ in range(60)]

    # # add 40 optional notes
    # e += [
    #     ec().intersect({"instrument": {"Bass", "Drums", "Piano", "-"}})
    #     for _ in range(40)
    # ]

    # pad with empty notes
    e += [ec().force_inactive() for _ in range(n_events - len(e))]

    # set to 125
    e = [ev.intersect(ec().tempo_constraint(110)) for ev in e]

    # set tag to pop
    e = [ev.intersect({"tag": {"pop", "-"}}) for ev in e]

    return e


# def band_beat(
#     e,
#     ec,
#     n_events,
#     beat_range,
#     pitch_range,
#     drums,
#     tag,
#     tempo,
# ):
#     # add
#     e = []
#     # add 20 bass
#     e += [ec().intersect({"instrument": {"Bass"}}).force_active() for _ in range(20)]
#     # add 40 piano
#     e += [ec().intersect({"instrument": {"Piano"}}).force_active() for _ in range(40)]

#     # add 70 drums
#     e += [ec().intersect({"instrument": {"Drums"}}).force_active() for _ in range(60)]

#     # add 40 optional notes
#     e += [
#         ec().intersect({"instrument": {"Bass", "Drums", "Piano", "-"}})
#         for _ in range(40)
#     ]

#     # pad with empty notes
#     e += [ec().force_inactive() for _ in range(n_events - len(e))]

#     # set to 125
#     e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]

#     # set tag to pop
#     e = [ev.intersect({"tag": {tag, "-"}}) for ev in e]

#     return e