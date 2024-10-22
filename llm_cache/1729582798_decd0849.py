# Add a bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a bass line to the existing loop.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Add bass notes on the first beat of each bar
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(bar * 4)},
                                        "pitch": {str(pitch) for pitch in range(30, 50)},  # Low to mid-range bass notes
                                    })
                                    .force_active()
                                ]
                            
                            # Add some additional bass notes
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in range(30, 50)},
                                    })
                                ]
                            
                            # Set velocity for bass notes
                            e = [ev.intersect(ec().velocity_constraint(80)) if ev.a["instrument"] == {"Bass"} else ev for ev in e]
                            
                            # Preserve the existing tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the existing tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to maintain the total number of events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e