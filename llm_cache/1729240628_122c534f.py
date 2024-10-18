# trap hip hop beat with rattling hats and 808s (35 midi pitch)
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a trap hip hop beat with rattling hats and 808s, we'll focus on the following elements:
                            - Heavy 808 bass (pitch 35)
                            - Snares or claps
                            - Kick drums
                            - Rattling hi-hats (both closed and open)
                            - Optional percussion elements
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add 808 bass (pitch 35)
                            e += [ec().intersect({"pitch": {"35 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add kicks
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add snares/claps on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)", "39 (Drums)"}})  # Snare or clap
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add rattling hi-hats (closed)
                            e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(32)]
                            
                            # Add some open hi-hats for variation
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add some additional percussion elements (optional)
                            e += [ec().intersect({"pitch": {"37 (Drums)", "43 (Drums)", "47 (Drums)"}}) for _ in range(10)]
                            
                            # Set tempo to 140 BPM (typical for trap)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
                            
                            # Set tag to hip-hop
                            e = [ev.intersect({"tag": {"hip-hop", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e