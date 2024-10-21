# Write some piano chords
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a piano chord progression, we'll add a series of chords 
                            that complement the existing funk beat and bassline.
                            '''
                            # Remove any existing piano notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Define chord structures (root positions)
                            chords = [
                                [0, 4, 7],  # Major triad
                                [0, 3, 7],  # Minor triad
                                [0, 4, 7, 10],  # Dominant 7th
                                [0, 3, 7, 10],  # Minor 7th
                            ]
                            
                            # Add piano chords on each beat of the 4 bars
                            for bar in range(4):
                                for beat in range(4):
                                    chord = chords[bar % len(chords)]  # Cycle through chord types
                                    root = 60 + (bar * 2) % 12  # Change root note every two bars
                                    
                                    for note in chord:
                                        e += [
                                            ec()
                                            .intersect({
                                                "instrument": {"Piano"},
                                                "pitch": {str(root + note)},
                                                "onset/beat": {str(bar * 4 + beat)},
                                                "offset/beat": {str(bar * 4 + beat + 1)},  # Hold for one beat
                                                "velocity": {"80", "90", "100"}  # Medium-loud
                                            })
                                            .force_active()
                                        ]
                            
                            # Add some optional piano notes for fills or embellishments
                            e += [ec().intersect({"instrument": {"Piano"}}) for _ in range(10)]
                            
                            # Ensure the funk tag is still applied
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Set tempo to 96 (typical for funk)
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e