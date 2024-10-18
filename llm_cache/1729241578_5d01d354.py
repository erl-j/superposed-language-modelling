# simple rock beat with a swing feel
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a simple rock beat with a swing feel, we'll focus on kick, snare, and hi-hat patterns with a slight emphasis on the offbeats to create the swing.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Basic rock pattern: kick on 1 and 3, snare on 2 and 4
                            for bar in range(4):
                                # Kick on 1 and 3
                                for beat in [0, 2]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                                # Snare on 2 and 4
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
            
                            # Hi-hat pattern with swing feel
                            for bar in range(4):
                                for beat in range(4):
                                    # On-beat hi-hat (slightly lower velocity)
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .intersect(ec().velocity_constraint(70))
                                        .force_active()
                                    ]
                                    # Off-beat hi-hat (slightly higher velocity for swing feel)
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat + bar * 4)}, "onset/tick": {"14"}})
                                        .intersect(ec().velocity_constraint(90))
                                        .force_active()
                                    ]
            
                            # Add some optional ghost notes for variety
                            e += [ec().intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)", "38 (Drums)", "42 (Drums)"}}) for _ in range(10)]
            
                            # Set tempo to 100 BPM (moderate tempo for rock with swing)
                            e = [ev.intersect(ec().tempo_constraint(100)) for ev in e]
            
                            # Set tags
                            e = [ev.intersect({"tag": {"rock", "swing", "-"}}) for ev in e]
            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e