# write a fast and frantic jungle drum beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a fast and frantic jungle drum beat. This will involve:
                            - A fast tempo
                            - Rapid kick drum patterns
                            - Syncopated snares
                            - Fast hi-hat patterns
                            - Some breakbeat-style drum fills
                            '''
                            e = []
                            # Set a fast tempo (around 170 BPM is typical for jungle)
                            tempo = 170
                            e = [ev.intersect(ec().tempo_constraint(170)) for ev in e]

                            # Rapid kick drum pattern
                            for beat in range(16):
                                if beat % 2 == 0:  # On every even beat
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                        .force_active()
                                    ]
                                if beat % 4 == 3:  # Additional kick on the 3rd 16th note of every beat
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"12"}})
                                        .force_active()
                                    ]

                            # Syncopated snares
                            for beat in [2, 6, 10, 14]:  # Typical jungle snare pattern
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]

                            # Fast hi-hat pattern
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                                # Add some open hi-hats for variation
                                if beat % 4 == 2:
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"46 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"12"}})
                                        .force_active()
                                    ]

                            # Add some breakbeat-style drum fills
                            for beat in [7, 15]:  # Add fills at the end of each 8-beat phrase
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                    for _ in range(3)  # Add 3 random drum hits for each fill
                                ]

                            # Add some ghost notes for additional complexity
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "velocity": {"30"}})
                                .force_active()
                                for _ in range(8)
                            ]

                            # Set tag to dance-eletric (closest to jungle in the available tags)
                            e = [ev.intersect({"tag": {"dance-eletric"}}) for ev in e]

                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]

                            return e