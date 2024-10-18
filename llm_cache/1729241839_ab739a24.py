# simple rock beat with a swing feel. ride triplets
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a simple rock beat with a swing feel and ride triplets, we'll focus on kick, snare, hi-hat, and ride cymbal.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Kick drum on beats 1 and 3 of each bar
                            for bar in range(4):
                                for beat in [0, 2]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"36 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Snare on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Hi-hat with swing feel
                            for bar in range(4):
                                for beat in range(4):
                                    # On-beat hi-hat
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                                    # Off-beat hi-hat (slightly delayed for swing feel)
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}, "onset/tick": {"14"}})
                                        .force_active()
                                    ]
                            
                            # Ride cymbal triplets
                            for bar in range(4):
                                for beat in range(4):
                                    for tick in [0, 8, 16]:
                                        e += [
                                            ec()
                                            .intersect({"pitch": {"51 (Drums)"}})
                                            .intersect({"onset/beat": {str(beat + bar * 4)}, "onset/tick": {str(tick)}})
                                            .force_active()
                                        ]
                            
                            # Add some optional drum hits for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo to 100 BPM (moderate rock tempo)
                            e = [ev.intersect(ec().tempo_constraint(100)) for ev in e]
                            
                            # Set tag to rock and swing
                            e = [ev.intersect({"tag": {"rock", "swing"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e