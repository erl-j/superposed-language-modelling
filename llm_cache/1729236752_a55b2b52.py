# drill rap beat tresillo rythm
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a drill rap beat with a tresillo rhythm, we'll focus on a heavy 808 bass, punchy kicks, snares, and hi-hats. The tresillo rhythm (3-3-2) will be incorporated into the hi-hat pattern.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add 808 bass (using low tom as a substitute)
                            e += [ec().intersect({"pitch": {"45 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add kicks (4 per bar)
                            for bar in range(4):
                                for beat in range(4):
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"36 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add snares on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add hi-hats with tresillo rhythm (3-3-2 pattern)
                            tresillo_pattern = [0, 6, 12, 18, 0, 6, 12, 16, 20]
                            for bar in range(4):
                                for tick in tresillo_pattern:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}})
                                        .intersect({"onset/beat": {str(bar * 4)}, "onset/tick": {str(tick)}})
                                        .force_active()
                                    ]
                            
                            # Add some open hi-hats for variation
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add up to 20 optional drum notes for additional complexity
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tempo to 140 BPM (typical for drill rap)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
                            
                            # Set tag to electronic (closest to drill rap in the given list)
                            e = [ev.intersect({"tag": {"electronic", "-"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events