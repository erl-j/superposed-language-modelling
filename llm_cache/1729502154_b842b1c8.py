# Write some calm piano chords.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a calm piano chord progression, we'll use a slow tempo, 
                            longer note durations, and softer velocities. We'll create a simple 
                            four-chord progression that repeats throughout the 4 bars.
                            '''
                            e = []
                            # Set a slower tempo for a calmer feel
                            tempo = 72
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Define our chord progression (using middle C as reference)
                            chords = [
                                [60, 64, 67],  # C major
                                [62, 65, 69],  # D minor
                                [65, 69, 72],  # F major
                                [67, 71, 74],  # G major
                            ]
                            
                            # Place chords on each downbeat
                            for bar in range(4):
                                for note in chords[bar % 4]:
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "pitch": {str(note) + " (Piano)"},
                                            "onset/beat": {str(bar * 4)},
                                            "offset/beat": {str(bar * 4 + 3)},  # Hold for 3 beats
                                        })
                                        .intersect(ec().velocity_constraint(70))  # Softer velocity
                                        .force_active()
                                    )
                            
                            # Add some optional notes for variation
                            e += [ec().intersect({"instrument": {"Piano"}}) for _ in range(10)]
                            
                            # Set tag to newage for a calmer feel
                            e = [ev.intersect({"tag": {"newage", "-"}}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e