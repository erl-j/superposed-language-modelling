# Write a funky bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a funky bass line, we'll create a pattern that emphasizes syncopation,
                            uses a limited range of notes, and includes some characteristic funk techniques.
                            '''
                            e = []
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define a funky bass pattern over two bars
                            pattern = [
                                (0, 0), (0, 12), (1, 0), (1, 12), (1, 18),  # Bar 1
                                (2, 6), (2, 18), (3, 0), (3, 12), (3, 18),  # Bar 2
                            ]
                            
                            # Repeat the pattern for 4 bars
                            for bar in range(0, 4, 2):
                                for beat, tick in pattern:
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/beat": {str(beat + bar * 4)},
                                            "onset/tick": {str(tick)},
                                            "pitch": {"36", "38", "40", "41", "43", "45"},  # E2 to A2 range
                                        })
                                        .force_active()
                                    )
                            
                            # Add some sliding notes (characteristic of funk bass)
                            slides = [(4, 0), (8, 0), (12, 0)]  # At the start of bars 2, 3, and 4
                            for beat, tick in slides:
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(beat)},
                                        "onset/tick": {str(tick)},
                                        "pitch": {"36", "38", "40", "41", "43", "45"},
                                    })
                                    .intersect(ec().velocity_constraint(100))  # Slightly louder for emphasis
                                    .force_active()
                                )
                            
                            # Add some ghost notes (very quiet notes for rhythm)
                            ghost_notes = [(1, 6), (3, 6), (5, 6), (7, 6), (9, 6), (11, 6), (13, 6), (15, 6)]
                            for beat, tick in ghost_notes:
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(beat)},
                                        "onset/tick": {str(tick)},
                                        "pitch": {"36", "38", "40", "41", "43", "45"},
                                    })
                                    .intersect(ec().velocity_constraint(30))  # Very quiet
                                    .force_active()
                                )
                            
                            # Set tempo to a funky 110 BPM
                            e = [ev.intersect(ec().tempo_constraint(110)) for ev in e]
                            
                            # Set tag to funk
                            e = [ev.intersect({"tag": {"funk"}}) for ev in e]
                            
                            # Add some optional bass notes for variation
                            e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(10)]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e