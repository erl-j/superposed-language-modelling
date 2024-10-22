# Create a 909 dance beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a classic 909 dance beat with kick, snare, hi-hats, and some percussion.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add kick drum on every quarter note
                            for beat in range(0, 16, 1):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Add snare on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add closed hi-hats on every eighth note
                            for beat in range(0, 16):
                                for tick in [0, 12]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {str(tick)}})
                                        .force_active()
                                    ]
                            
                            # Add open hi-hat occasionally
                            for _ in range(4):
                                e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active()]
                            
                            # Add some percussion (e.g., clap or cowbell)
                            for _ in range(8):
                                e += [ec().intersect({"pitch": {"39 (Drums)", "56 (Drums)"}}).force_active()]
                            
                            # Set a dance tempo (around 128 BPM)
                            e = [ev.intersect(ec().tempo_constraint(128)) for ev in e]
                            
                            # Set tag to dance-eletric
                            e = [ev.intersect({"tag": {"dance-eletric"}}) for ev in e]
                            
                            # Add some optional drum hits for variety
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e