# Add a bassline in the middle register
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a bassline in the middle register to the existing beat.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})}]
                            
                            # Add bass notes
                            for beat in range(16):  # Assuming a 4-bar loop (4 * 4 beats)
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(beat)},
                                        "pitch": {str(pitch) for pitch in range(48, 60)},  # Middle register (C3 to B3)
                                    })
                                    .force_active()
                                ]
                            
                            # Add some optional bass notes for variation
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(pitch) for pitch in range(48, 60)},
                                })
                                for _ in range(8)
                            ]
                            
                            # Preserve the tempo from the input
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the tag from the input
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e