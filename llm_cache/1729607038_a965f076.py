# add a repetitive guitar line with only 3 pitches
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a repetitive guitar line with only 3 pitches to the existing beat.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove any existing guitar notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Guitar"})]
                            
                            # Define the 3 pitches for the guitar line
                            guitar_pitches = {str(pitch) for pitch in range(60, 63)}  # Using E4, F4, F#4 as an example
                            
                            # Create a repetitive pattern for 4 bars (16 beats)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Guitar"},
                                        "onset/beat": {str(beat)},
                                        "pitch": guitar_pitches,
                                    })
                                    .force_active()
                                ]
                            
                            # Add some variation with optional notes
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Guitar"},
                                    "pitch": guitar_pitches,
                                })
                                for _ in range(8)
                            ]
                            
                            # Preserve the tempo from the existing beat
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the tag from the existing beat
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e