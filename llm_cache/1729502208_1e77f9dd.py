# Write some calm piano chords.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a calm piano chord progression, we'll use a slow tempo, 
                            longer note durations, and softer velocities. We'll create a simple 
                            progression using common chords in a major key.
                            '''
                            e = []
                            # Set a calm tempo
                            tempo = 72
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Define some common chord structures (root positions)
                            chords = [
                                [60, 64, 67],  # C Major
                                [65, 69, 72],  # F Major
                                [67, 71, 74],  # G Major
                                [62, 65, 69],  # D Minor
                            ]
                            
                            # Place chords on each downbeat
                            for bar in range(4):
                                chord = chords[bar % len(chords)]
                                for note in chord:
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "pitch": {str(note)},
                                            "onset/beat": {str(bar * 4)},
                                            "offset/beat": {str(bar * 4 + 3)},  # Hold for 3 beats
                                            "velocity": {"60-80"},  # Soft to medium velocity
                                        })
                                        .force_active()
                                    )
                            
                            # Add some optional notes for variation
                            for _ in range(8):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "pitch": {"60-84"},  # C4 to C6
                                        "velocity": {"50-70"},  # Even softer
                                    })
                                )
                            
                            # Set tag to newage for a calm feel
                            e = [ev.intersect({"tag": {"newage", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e