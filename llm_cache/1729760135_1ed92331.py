# Add some more bass notes in a higher register
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add some higher register bass notes to the existing beat while keeping the original bass line.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Separate bass and other instruments
                            e_bass = [ev for ev in e if ev.a["instrument"].intersection({"Bass"})]
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Start with existing events
                            new_e = e_bass + e_other
                            
                            # Add new higher register bass notes
                            for _ in range(8):  # Add up to 8 new bass notes
                                new_e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in range(46, 60)},  # Higher register
                                    })
                                    .force_active()
                                )
                            
                            # Add some optional bass notes
                            new_e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(4)]
                            
                            # Preserve tempo
                            new_e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in new_e]
                            
                            # Preserve tag
                            new_e = [ev.intersect({"tag": {tag}}) for ev in new_e]
                            
                            # Pad with inactive notes
                            new_e += [ec().force_inactive() for _ in range(n_events - len(new_e))]
                            
                            return new_e