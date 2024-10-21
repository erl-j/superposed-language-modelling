# add a bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a funky bass line that complements the existing drum pattern.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define bass line pattern
                            bass_pattern = [
                                (0, 0), (0, 12), (1, 0), (1, 12), # First beat
                                (2, 0), (2, 12), (3, 0), (3, 12), # Second beat
                                (4, 0), (4, 12), (5, 0), (5, 12), # Third beat
                                (6, 0), (6, 12), (7, 0), (7, 12)  # Fourth beat
                            ]
                            
                            # Add bass notes
                            for beat, tick in bass_pattern:
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(beat)},
                                        "onset/tick": {str(tick)},
                                        "pitch": {str(pitch) for pitch in range(35, 55)},  # E1 to G2
                                        "offset/beat": {str(min(beat + 1, 15))},  # Ensure note doesn't go beyond 4 bars
                                        "offset/tick": {str(min(tick + 12, 23))}  # Note duration of half a beat
                                    })
                                    .force_active()
                                ]
                            
                            # Add some variation
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(pitch) for pitch in range(35, 55)}
                                })
                                for _ in range(8)  # Add up to 8 additional bass notes for variation
                            ]
                            
                            # Set velocity for bass notes
                            e = [ev.intersect(ec().velocity_constraint(80)) if ev.a["instrument"] == {"Bass"} else ev for ev in e]
                            
                            # Maintain current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag
                            e = [ev.intersect({"tag": {"funk"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e