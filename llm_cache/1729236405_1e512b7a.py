# dense hip-hop beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a dense hip-hop beat, we'll use a combination of kicks, snares, hi-hats, and additional percussion elements.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add a strong kick pattern (12 kicks)
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(12)]
                            
                            # Add snares on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add a dense hi-hat pattern (16 closed hi-hats)
                            e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(16)]
                            
                            # Add some open hi-hats for variation (4 open hi-hats)
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add some percussion elements for additional texture
                            # Clap (3 instances)
                            e += [ec().intersect({"pitch": {"39 (Drums)"}}).force_active() for _ in range(3)]
                            # Cowbell (2 instances)
                            e += [ec().intersect({"pitch": {"56 (Drums)"}}).force_active() for _ in range(2)]
                            # Tom (3 instances)
                            e += [ec().intersect({"pitch": {"50 (Drums)"}}).force_active() for _ in range(3)]
                            
                            # Add up to 10 optional drum notes for even more density
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo to 90 BPM (typical for hip-hop)
                            e = [ev.intersect(ec().tempo_constraint(90)) for ev in e]
                            
                            # Set tag to hip-hop
                            e = [ev.intersect({"tag": {"hip-hop", "-"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events