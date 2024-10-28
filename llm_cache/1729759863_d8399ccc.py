# Write a house beat with open hihats on the offbeats
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a house beat with a typical four-on-the-floor kick pattern,
                            snares on the 2 and 4, closed hi-hats on every beat, and open hi-hats on the offbeats.
                            '''
                            e = []
                            # Set the kick drum (four-on-the-floor pattern)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Set the snare on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Set closed hi-hats on every beat
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Set open hi-hats on the offbeats
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"46 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
                            
                            # Add some optional percussion elements
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo to a typical house tempo (128 BPM)
                            e = [ev.intersect(ec().tempo_constraint(128)) for ev in e]
                            
                            # Set tag to house
                            e = [ev.intersect({"tag": {"house"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e