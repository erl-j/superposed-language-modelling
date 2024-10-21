# Add a funky bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a funky bass line, we'll create a pattern that emphasizes syncopation,
                            uses a range of notes typical for funk, and includes some rhythmic variations.
                            '''
                            # Remove existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define a funky bass pattern (2 bars, repeated twice)
                            funky_pattern = [
                                (0, 0, 2), (0, 12, 1),  # Beat 1
                                (1, 0, 1), (1, 12, 1),  # Beat 2
                                (2, 0, 2), (2, 18, 1),  # Beat 3
                                (3, 6, 1), (3, 18, 1),  # Beat 4
                                (4, 0, 2), (4, 12, 1),  # Beat 5
                                (5, 0, 1), (5, 12, 1),  # Beat 6
                                (6, 0, 1), (6, 12, 2),  # Beat 7
                                (7, 6, 1), (7, 18, 1),  # Beat 8
                            ]
                            
                            # Repeat the pattern for 4 bars
                            for bar in range(2):
                                for beat, tick, duration in funky_pattern:
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/beat": {str(beat + bar * 8)},
                                            "onset/tick": {str(tick)},
                                            "offset/beat": {str(beat + bar * 8 + duration)},
                                        })
                                        .intersect(ec().pitch_constraint(range(36, 48)))  # E1 to B1, typical funk bass range
                                        .intersect(ec().velocity_constraint(80, 110))  # Medium-high velocity for funk
                                        .force_active()
                                    )
                            
                            # Add some variation notes
                            for _ in range(5):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                    })
                                    .intersect(ec().pitch_constraint(range(36, 48)))
                                    .intersect(ec().velocity_constraint(70, 100))
                                )
                            
                            # Set funk tag
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Set tempo to a typical funk tempo
                            e = [ev.intersect(ec().tempo_constraint(110)) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e