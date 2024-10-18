# reggae beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a reggae beat, we'll focus on the characteristic one-drop rhythm, 
                            with emphasis on the 2nd and 4th beats, and incorporate some typical reggae percussion elements.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Kick drum on beats 3 of each bar (one-drop rhythm)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(2 + bar * 4)}})
                                    .force_active()
                                ]
                            
                            # Snare (or rim shot) on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Hi-hat pattern (closed)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Occasional open hi-hat
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add some percussion elements typical in reggae (e.g., congas, bongos)
                            e += [ec().intersect({"pitch": {"60 (Drums)", "61 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add some optional drum notes for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo to a typical reggae tempo (60-90 BPM)
                            e = [ev.intersect(ec().tempo_constraint(70)) for ev in e]
                            
                            # Set tag to reggae
                            e = [ev.intersect({"tag": {"reggae", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e