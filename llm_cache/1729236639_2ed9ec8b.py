# drill rap beat dotted eight notes hi-hats
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a drill rap beat with dotted eighth note hi-hats, we'll create a pattern with kicks, snares, hi-hats, and some optional percussion elements.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add kicks (typically on the 1 and in between 2 and 3)
                            e += [ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {"0"}}).force_active()]
                            e += [ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {"2"}, "onset/tick": {"12"}}).force_active()]
                            
                            # Add snares (typically on 2 and 4)
                            e += [ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {"4", "12"}}).force_active()]
                            
                            # Add dotted eighth note hi-hats
                            for bar in range(4):
                                for beat in range(4):
                                    onset_beat = bar * 4 + beat
                                    e += [ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(onset_beat)}, "onset/tick": {"0"}}).force_active()]
                                    e += [ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(onset_beat)}, "onset/tick": {"18"}}).force_active()]
            
                            # Add some open hi-hats for variation
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add some 808 bass notes (common in drill beats)
                            e += [ec().intersect({"pitch": {"35 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add some optional percussion elements
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tempo to 140 BPM (common for drill rap)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
                            
                            # Set tag to a combination of relevant styles
                            e = [ev.intersect({"tag": {"electronic", "hip-hop", "urban", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e