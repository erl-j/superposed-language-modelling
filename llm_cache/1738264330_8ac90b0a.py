# add some jangly guitar
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a jangly guitar to the existing beat. Jangly guitar typically involves bright, ringing chords or arpeggios.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside existing instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Guitar"})]
                            
                            # Start with existing events
                            e = e_other.copy()
                            
                            # Add jangly guitar parts
                            # We'll add some strummed chords and arpeggios
                            
                            # Strummed chords (4 instances)
                            for _ in range(4):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Guitar"},
                                        "pitch": {str(pitch) for pitch in range(60, 85)},  # Higher pitches for brightness
                                        "duration": {"1/4", "1/2"}  # Longer durations for ringing effect
                                    })
                                    .intersect(ec().velocity_constraint(80))  # Moderately loud
                                    .force_active()
                                ]
                            
                            # Arpeggios (8 instances)
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Guitar"},
                                        "pitch": {str(pitch) for pitch in range(55, 90)},  # Wide range for arpeggios
                                        "duration": {"1/16", "1/8"}  # Shorter durations for arpeggio notes
                                    })
                                    .intersect(ec().velocity_constraint(70))  # Slightly softer than chords
                                    .force_active()
                                ]
                            
                            # Add some optional guitar notes for variety
                            e += [ec().intersect({"instrument": {"Guitar"}}) for _ in range(10)]
                            
                            # Preserve tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e