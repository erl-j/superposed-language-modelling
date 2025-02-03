# write a simple drum beat with many kicks
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a simple drum beat with many kicks, some snares, and hi-hats.
                            '''
                            e = []
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add many kicks (20 in total)
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(20)]
                            
                            # Add snares on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}})
                                        .intersect({"onset/global_tick": {str(beat * 24 + bar * 96)}})
                                        .force_active()
                                    ]
                            
                            # Add hi-hats (16 in total)
                            e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(16)]
                            
                            # Add up to 10 optional drum notes
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo (assuming a moderate tempo of 120 BPM)
                            e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]
                            
                            # Set tag to 'other' as it's a simple beat
                            e = [ev.intersect({"tag": {"other"}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e