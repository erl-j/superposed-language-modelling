# give me a techno beat with open hats
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a techno beat with open hats, we'll use a four-on-the-floor kick pattern, 
                            snares on the off-beats, closed hi-hats, and emphasize open hi-hats.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Four-on-the-floor kick pattern
                            for beat in range(0, 16, 4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Snares on off-beats (2 and 4)
                            for beat in range(4, 16, 4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Closed hi-hats on every beat
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Open hi-hats every other beat, emphasizing the off-beats
                            for beat in range(1, 16, 2):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"46 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Add some variation with additional percussion
                            e += [ec().intersect({"pitch": {"39 (Drums)", "40 (Drums)", "41 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add up to 20 optional drum notes for further variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tempo to 128 BPM (typical for techno)
                            e = [ev.intersect(ec().tempo_constraint(128)) for ev in e]
                            
                            # Set tag to techno
                            e = [ev.intersect({"tag": {"techno", "-"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events