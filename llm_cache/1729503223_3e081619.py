# Write some sad piano chords.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a sad piano chord progression, we'll use minor chords and a slow tempo.
                            We'll create a sequence of four chords, each lasting a full bar.
                            '''
                            # Remove any existing piano notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Define sad chord progression (using minor chords)
                            # We'll use Am, Dm, Em, Am as our progression
                            chord_progression = [
                                [57, 60, 64],  # Am
                                [62, 65, 69],  # Dm
                                [64, 67, 71],  # Em
                                [57, 60, 64],  # Am
                            ]
                            
                            # Add chords
                            for bar, chord in enumerate(chord_progression):
                                for note in chord:
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "pitch": {str(note)},
                                            "onset/beat": {str(bar * 4)},
                                            "offset/beat": {str((bar + 1) * 4)},
                                        })
                                        .force_active()
                                    )
                            
                            # Set a slow tempo for a sadder feel
                            e = [ev.intersect(ec().tempo_constraint(60)) for ev in e]
                            
                            # Set tag to classical or romantic for a more emotional feel
                            e = [ev.intersect({"tag": {"classical", "romantic"}}) for ev in e]
                            
                            # Add some optional piano notes for potential embellishments
                            e += [ec().intersect({"instrument": {"Piano"}}) for _ in range(10)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e