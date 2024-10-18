# hip-hop beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a hip-hop beat, we'll focus on a strong kick and snare pattern, 
                            with hi-hats and some additional percussion elements.
                            '''
                            e = []
                            # Clear existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Kick drum pattern (typically on 1 and 3, with some variations)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(1 + bar * 4)}}).force_active(),
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(3 + bar * 4)}}).force_active()
                                ]
                            # Add 4 more optional kicks for variation
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}) for _ in range(4)]
                            
                            # Snare drum pattern (typically on 2 and 4)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(2 + bar * 4)}}).force_active(),
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(4 + bar * 4)}}).force_active()
                                ]
                            
                            # Hi-hat pattern (closed hi-hat)
                            e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(16)]
                            
                            # Open hi-hat (occasional)
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}) for _ in range(4)]
                            
                            # Additional percussion (e.g., clap, rim shot)
                            e += [ec().intersect({"pitch": {"39 (Drums)", "37 (Drums)"}}) for _ in range(8)]
                            
                            # Set tempo to a typical hip-hop range (85-95 BPM)
                            tempo = 90
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to hip-hop
                            e = [ev.intersect({"tag": {"hip-hop", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e