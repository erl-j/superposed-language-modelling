# Add some more open hi-hats
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add more open hi-hats to the existing drum pattern.
                            '''
                            # Keep all existing events
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Add 6 more open hi-hats
                            for _ in range(6):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Drums"},
                                        "pitch": {"46 (Drums)"}, # 46 is the MIDI pitch for open hi-hat
                                    })
                                    .force_active()
                                ]
                            
                            # Add some variation in velocity for the new hi-hats
                            for ev in e[-6:]:
                                ev = ev.intersect(ec().velocity_constraint(80))  # Set a moderate velocity
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to maintain the total number of events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e