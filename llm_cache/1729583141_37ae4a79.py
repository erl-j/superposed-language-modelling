# add some 909 drums
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add 909-style drums to the existing loop. The 909 is known for its punchy kick, 
                            crisp snare, and distinctive hi-hats. We'll preserve the existing elements and add these new drum sounds.
                            '''
                            # Preserve existing events
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Add 909-style kick drum (36 is typically used for kick)
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(beat)} for beat in [0, 4, 8, 12]})
                                .intersect(ec().velocity_constraint(100))  # 909 kicks are usually loud
                                .force_active()
                                for _ in range(4)
                            ]
                            
                            # Add 909-style snare (40 is often used for 909 snare)
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {"40 (Drums)"}, "onset/beat": {str(beat)} for beat in [2, 6, 10, 14]})
                                .intersect(ec().velocity_constraint(90))
                                .force_active()
                                for _ in range(4)
                            ]
                            
                            # Add 909-style closed hi-hats (42 for closed hi-hat)
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/beat": {str(beat)} for beat in range(16)})
                                .intersect(ec().velocity_constraint(80))
                                .force_active()
                                for _ in range(16)
                            ]
                            
                            # Add 909-style open hi-hats (46 for open hi-hat)
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {"46 (Drums)"}, "onset/beat": {str(beat)} for beat in [3, 7, 11, 15]})
                                .intersect(ec().velocity_constraint(85))
                                .force_active()
                                for _ in range(4)
                            ]
                            
                            # Add some optional percussion hits (e.g., 909 clap or tom)
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {"39 (Drums)", "41 (Drums)", "43 (Drums)", "45 (Drums)"}, "onset/beat": {str(beat) for beat in range(16)}})
                                .intersect(ec().velocity_constraint(75))
                                for _ in range(8)
                            ]
                            
                            # Preserve the existing tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the existing tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to reach n_events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e