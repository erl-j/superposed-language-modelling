# write a simple drum beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a simple drum beat with kick, snare, and hi-hat.
                            '''
                            e = []
                            # Set kick drum on beats 1 and 3
                            for bar in range(4):
                                for beat in [0, 2]:
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/global_tick": {str(beat * 24 + bar * 96)}})
                                        .force_active()
                                    ]
                            
                            # Set snare on beats 2 and 4
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/global_tick": {str(beat * 24 + bar * 96)}})
                                        .force_active()
                                    ]
                            
                            # Set hi-hat on every eighth note
                            for bar in range(4):
                                for eighth in range(8):
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/global_tick": {str(eighth * 12 + bar * 96)}})
                                        .force_active()
                                    ]
                            
                            # Set tempo (assuming a moderate tempo of 120 BPM)
                            e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]
                            
                            # Set tag (assuming a general "pop" tag)
                            e = [ev.intersect({"tag": {"pop"}}) for ev in e]
                            
                            # Add some optional drum hits for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e