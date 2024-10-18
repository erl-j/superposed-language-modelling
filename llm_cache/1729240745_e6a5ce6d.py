# beat with rattling hats and 808s (35 midi pitch). Dense and fast and syncopated
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a dense, fast, and syncopated beat with rattling hats and 808s.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add 808 kicks (35 midi pitch)
                            e += [ec().intersect({"pitch": {"35 (Drums)"}}).force_active() for _ in range(16)]
                            
                            # Add rattling hi-hats (closed and open)
                            e += [ec().intersect({"pitch": {"42 (Drums)", "46 (Drums)"}}).force_active() for _ in range(32)]
                            
                            # Add some snares for variety
                            e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add syncopated elements
                            syncopated_beats = ["0/12", "1/12", "2/12", "3/12", "0/18", "1/18", "2/18", "3/18"]
                            for beat in syncopated_beats:
                                e += [
                                    ec()
                                    .intersect({"onset/beat": {beat.split('/')[0]}})
                                    .intersect({"onset/tick": {beat.split('/')[1]}})
                                    .intersect({"pitch": {"35 (Drums)", "42 (Drums)", "46 (Drums)"}})
                                    .force_active()
                                ]
                            
                            # Add some ghost notes for additional density
                            e += [
                                ec()
                                .intersect({"pitch": {"35 (Drums)", "38 (Drums)", "42 (Drums)"}})
                                .intersect(ec().velocity_constraint(40))
                                .force_active()
                                for _ in range(16)
                            ]
                            
                            # Set a fast tempo (160 BPM)
                            e = [ev.intersect(ec().tempo_constraint(160)) for ev in e]
                            
                            # Set appropriate tags
                            e = [ev.intersect({"tag": {"electronic", "breakbeat", "fast", "-"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e