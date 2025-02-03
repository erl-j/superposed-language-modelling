# add two piano chords
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add two piano chords to the existing beat.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside existing instruments
                            e_existing = e.copy()
                            
                            # Start with an empty list for new events
                            e = []
                            
                            # Add two piano chords
                            for i in range(2):
                                chord = ec().intersect({
                                    "instrument": {"Piano"},
                                    "duration": {"1/2", "1"}, # Half note or whole note duration
                                    "onset/global_tick": {str(i * 192)}, # Place chords at the start of bar 1 and 3
                                }).force_active()
                                
                                # Add three notes to form a chord
                                for _ in range(3):
                                    e.append(chord.copy())
            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back existing instruments
                            e += e_existing
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e