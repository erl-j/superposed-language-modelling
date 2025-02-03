# write a jungle drum beat
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a jungle drum beat with fast breakbeats, syncopated rhythms, and a mix of acoustic and electronic drum sounds.
                            '''
                            e = []
                            # Set a fast tempo typical for jungle
                            tempo = 170
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

                            # Kick drum pattern (alternating between acoustic and electronic)
                            for i in range(8):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"} if i % 2 == 0 else {"35 (Drums)"}, "onset/global_tick": {str(i * 48)}})
                                    .force_active()
                                ]

                            # Snare on the 2 and 4, with some ghost notes
                            for i in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}, "onset/global_tick": {str(i * 96 + 48)}})
                                    .force_active()
                                ]
                                # Ghost notes
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}, "onset/global_tick": {str(i * 96 + 24)}})
                                    .intersect(ec().velocity_constraint(40))
                                    .force_active()
                                ]

                            # Hi-hats (closed and open) for a rolling rhythm
                            for i in range(32):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"} if i % 4 != 3 else {"46 (Drums)"}, "onset/global_tick": {str(i * 12)}})
                                    .force_active()
                                ]

                            # Add some tom fills
                            tom_pitches = ["45 (Drums)", "47 (Drums)", "50 (Drums)"]
                            for i in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {tom_pitches[i % 3]}, "onset/global_tick": {str(336 + i * 12)}})
                                    .force_active()
                                ]

                            # Add some percussion hits (e.g., cowbell or clave) for flavor
                            for i in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"56 (Drums)"}, "onset/global_tick": {str(i * 96 + 72)}})
                                    .force_active()
                                ]

                            # Allow for some additional drum hits to add complexity
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]

                            # Set tag to dance-eletric (closest to jungle)
                            e = [ev.intersect({"tag": {"dance-eletric"}}) for ev in e]

                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]

                            return e