# add some 909 drums
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add 909-style drums to the existing beat. The 909 is known for its punchy kick, 
                            crisp snare, and distinctive hi-hats. We'll add these elements while preserving the existing beat.
                            '''
                            # Preserve existing events
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Add 909-style kick drum (36 is typically used for kick)
                            e += [ec().intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add 909-style snare drum (40 is often used for 909 snare)
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {"40 (Drums)"}, "onset/beat": {str(beat)} })
                                .force_active()
                                for beat in [2, 6, 10, 14]  # Typical snare on 2 and 4 of each bar
                            ]
                            
                            # Add 909-style closed hi-hat (42 is typically used for closed hi-hat)
                            e += [ec().intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}}).force_active() for _ in range(16)]
                            
                            # Add 909-style open hi-hat (46 is typically used for open hi-hat)
                            e += [ec().intersect({"instrument": {"Drums"}, "pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add some 909-style tom fills (45, 47, 50 are often used for toms)
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {"45 (Drums)", "47 (Drums)", "50 (Drums)"}, "onset/beat": {"15"}})
                                .force_active()
                                for _ in range(3)  # Add a short fill at the end of the loop
                            ]
                            
                            # Preserve the existing tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the existing tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to reach n_events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e