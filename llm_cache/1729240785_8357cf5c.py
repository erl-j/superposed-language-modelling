# beat with rattling hats and 808s (35 midi pitch). Dense and fast and syncopated kicks
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a beat with rattling hats, 808s, dense and fast syncopated kicks.
                            '''
                            e = []
                            # remove all drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add dense, syncopated kicks (20 kicks)
                            for _ in range(20):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}})
                                    .intersect(ec().onset_constraint(0, 15, 0, 23))  # Allow kicks on any beat and tick
                                    .force_active()
                                ]
                            
                            # Add 808s (35 midi pitch) - 15 of them
                            for _ in range(15):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"35 (Drums)"}})
                                    .intersect(ec().onset_constraint(0, 15, 0, 23))  # Allow 808s on any beat and tick
                                    .force_active()
                                ]
                            
                            # Add rattling hats (30 closed hats, 10 open hats)
                            for _ in range(30):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}})  # Closed hi-hat
                                    .intersect(ec().onset_constraint(0, 15, 0, 23))  # Allow on any beat and tick
                                    .force_active()
                                ]
                            for _ in range(10):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"46 (Drums)"}})  # Open hi-hat
                                    .intersect(ec().onset_constraint(0, 15, 0, 23))  # Allow on any beat and tick
                                    .force_active()
                                ]
                            
                            # Add some snares for variety (8 snares)
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}})
                                    .intersect(ec().onset_constraint(0, 15, 0, 23))  # Allow on any beat and tick
                                    .force_active()
                                ]
                            
                            # Add up to 20 optional drum notes for additional variety
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set a fast tempo (140 BPM)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
                            
                            # Set tag to electronic and driving
                            e = [ev.intersect({"tag": {"electronic", "driving"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events