# Create a simple bassline
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a simple bassline that complements the existing beat.
                            '''
                            # Remove inactive notes and any existing bass notes
                            e = [ev for ev in e if ev.is_active() and ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Create a simple bassline
                            for bar in range(4):  # Assuming a 4-bar loop
                                # Add a bass note on the first beat of each bar
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(bar * 4)},  # First beat of each bar
                                        "pitch": {str(pitch) for pitch in range(36, 48)},  # E1 to B1
                                    })
                                    .force_active()
                                )
                                
                                # Add an optional bass note on the third beat of each bar
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(bar * 4 + 2)},  # Third beat of each bar
                                        "pitch": {str(pitch) for pitch in range(36, 48)},  # E1 to B1
                                    })
                                )
            
                            # Add up to 4 more optional bass notes for variation
                            for _ in range(4):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in range(36, 48)},  # E1 to B1
                                    })
                                )
            
                            # Preserve the tempo from the input events
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the tag from the input events
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to reach n_events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e