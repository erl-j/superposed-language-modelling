# add some short guitar stabs
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add short guitar stabs to the existing beat.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside existing instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Guitar"})]
                            
                            # Start with existing events
                            e = e_other.copy()
                            
                            # Add short guitar stabs
                            for _ in range(8):  # Add up to 8 guitar stabs
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Guitar"},
                                        "duration": {"1/16", "1/8"},  # Short durations for stabs
                                        "pitch": {str(pitch) for pitch in range(50, 70)}  # Mid-range pitches
                                    })
                                    .intersect(ec().velocity_constraint(80))  # Moderately loud
                                    .force_active()
                                ]
                            
                            # Add a few optional guitar notes
                            e += [ec().intersect({"instrument": {"Guitar"}}) for _ in range(4)]
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e