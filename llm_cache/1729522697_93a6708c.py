# add a bass line with current tag
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a bass line that complements the existing funk beat, 
                            while maintaining the current tag and tempo.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Find kicks to lock in with
                            kicks = [ev for ev in e if {"35 (Drums)", "36 (Drums)", "37 (Drums)"}.intersection(ev.a["pitch"])]
                            
                            # Add bass notes that lock in with kicks
                            for kick in kicks:
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": kick.a["onset/beat"],
                                        "onset/tick": kick.a["onset/tick"],
                                        "pitch": {str(pitch) for pitch in range(30, 55)},  # Bass range
                                    })
                                    .force_active()
                                )
                            
                            # Add some additional bass notes for variation
                            for _ in range(8):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in range(30, 55)},  # Bass range
                                    })
                                )
                            
                            # Add some longer bass notes
                            for _ in range(4):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in range(30, 55)},  # Bass range
                                    })
                                    .intersect(ec().min_duration_constraint(2))  # At least 2 beats long
                                )
                            
                            # Set velocity for bass notes
                            e = [ev.intersect(ec().velocity_constraint(80)) if ev.a["instrument"] == {"Bass"} else ev for ev in e]
                            
                            # Maintain current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Maintain current tag
                            e = [ev.intersect({"tag": tag}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e