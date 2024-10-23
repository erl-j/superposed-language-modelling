# write a fast drum n bass drum beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a fast drum and bass beat with a typical breakbeat pattern,
                            featuring kicks, snares, and hi-hats. We'll aim for a high tempo and complex rhythms.
                            '''
                            e = []
                            # Set a fast tempo typical for drum and bass (around 170-180 BPM)
                            tempo = 175
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

                            # Create a basic breakbeat pattern
                            # Kick drum (usually on the 1 and sometime on the 3)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}})
                                    .force_active()
                                ]
                                # Sometimes add a kick on the 3
                                if bar % 2 == 0:
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}})
                                        .force_active()
                                    ]

                            # Snare (usually on the 2 and 4)
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + beat)}})
                                        .force_active()
                                    ]

                            # Hi-hats (16th note pattern)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"0", "6", "12", "18"}})
                                    .force_active()
                                ]

                            # Add some ghost notes and variations
                            e += [ec().intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)", "40 (Drums)", "42 (Drums)"}}).intersect(ec().velocity_constraint(40)) for _ in range(10)]

                            # Add some cymbal crashes
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {"49 (Drums)"}, "onset/beat": {"0", "8"}})
                                .force_active()
                                for _ in range(2)
                            ]

                            # Allow for some additional drum hits for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]

                            # Set tag to dance-eletric (closest to drum and bass)
                            e = [ev.intersect({"tag": {"dance-eletric"}}) for ev in e]

                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]

                            return e