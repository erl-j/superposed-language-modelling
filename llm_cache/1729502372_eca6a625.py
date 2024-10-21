# Write some piano chords
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a piano chord progression, we'll add a series of chords 
                            that complement the existing funk groove. We'll use common funk chord voicings.
                            '''
                            # Remove any existing piano notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Define some funk chord voicings (root position 7th chords)
                            chord_voicings = [
                                [60, 64, 67, 70],  # C7
                                [62, 66, 69, 72],  # D7
                                [65, 69, 72, 75],  # F7
                                [67, 71, 74, 77],  # G7
                            ]
                            
                            # Add chords on beats 1 and 3 of each bar
                            for bar in range(4):
                                for beat in [0, 2]:
                                    chord = chord_voicings[bar % len(chord_voicings)]
                                    for note in chord:
                                        e.append(
                                            ec()
                                            .intersect({
                                                "instrument": {"Piano"},
                                                "pitch": {str(note)},
                                                "onset/beat": {str(bar * 4 + beat)},
                                                "offset/beat": {str(bar * 4 + beat + 2)},  # Hold for 2 beats
                                            })
                                            .force_active()
                                        )
                            
                            # Add some optional piano fills
                            for _ in range(10):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "pitch": pitch_range,
                                    })
                                )
                            
                            # Ensure all piano notes have appropriate velocity
                            for ev in e:
                                if ev.a["instrument"].issubset({"Piano"}):
                                    ev.intersect(ec().velocity_constraint(70, 100))
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e