# now keep everything the same but add some midi pitch 35 drum notes (808 bass)
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a funk beat with snares on the 2 and 4, triplets in the last bar, and added 808 bass (MIDI pitch 35).
                            '''
                            e = []
                            # remove all drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            # add 10 kicks
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(10)]
                            # snares on the 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            # add 10 hihats
                            e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(10)]
                            # add 4 open hihats
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
                            # add 10 ghost snare
                            e += [
                                ec()
                                .intersect({"pitch": {"38 (Drums)"}})
                                .intersect(ec().velocity_constraint(40))
                                .force_active()
                                for _ in range(10)
                            ]
                            # add 8 808 bass notes (MIDI pitch 35)
                            e += [ec().intersect({"pitch": {"35 (Drums)"}}).force_active() for _ in range(8)]
                            # add up to 20 optional drum notes
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            # add some triplets in the last bar
                            e += [ec().intersect({"onset/beat": {"12","13","14","15"}, "onset/tick": {"8", "16"}}).force_active() for _ in range(5)]
                            # set tempo to 96
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            # set tag to funk
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            # important: always pad with empty notes!
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            return e  # return the events