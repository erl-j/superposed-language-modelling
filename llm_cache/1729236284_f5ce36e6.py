# hip-hop beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a hip-hop beat, we'll focus on a strong, punchy kick, a snare on the 2 and 4, 
                            and a mix of hi-hats for rhythm. We'll also add some optional percussion for flavor.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add kicks (typically on 1 and 3, with some variations)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(1 + bar * 4)}}).force_active(),
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(3 + bar * 4)}}).force_active()
                                ]
                            # Add 2-3 more kicks for variation
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(3)]
                            
                            # Snares on 2 and 4
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(2 + bar * 4)}}).force_active(),
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(4 + bar * 4)}}).force_active()
                                ]
                            
                            # Hi-hats (mix of closed and open)
                            e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(16)]  # Closed hi-hats
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]   # Open hi-hats
                            
                            # Add some optional percussion (claps, toms, etc.)
                            e += [ec().intersect({"pitch": {"39 (Drums)", "41 (Drums)", "43 (Drums)", "45 (Drums)"}}) for _ in range(8)]
                            
                            # Set tempo to a typical hip-hop range (85-95 BPM)
                            e = [ev.intersect(ec().tempo_constraint(90)) for ev in e]
                            
                            # Set tag to hip-hop
                            e = [ev.intersect({"tag": {"hip-hop", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e