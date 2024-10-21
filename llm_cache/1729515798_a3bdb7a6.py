# Write a bass riff that goes with this drum beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a funky bass riff that complements the existing drum beat.
                            The bass riff will have a mix of longer and shorter notes, with some syncopation.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define our bass riff
                            bass_riff = [
                                # Bar 1
                                (0, 1, 43),  # Root note on beat 1
                                (1, 0.5, 46),  # Syncopated note on the "and" of 1
                                (2, 0.5, 43),  # Short note on 2
                                (2.5, 0.5, 41),  # Short note on the "and" of 2
                                (3, 1, 39),  # Longer note on 3
                                # Bar 2
                                (4, 0.75, 43),  # Root note again, slightly shorter
                                (5, 0.5, 46),  # Syncopated note
                                (5.75, 0.25, 43),  # Quick note
                                (6, 1, 41),  # Longer note on 2
                                (7, 0.5, 39),  # Short note on 3
                                (7.5, 0.5, 41),  # Short note on the "and" of 3
                                # Bar 3 (repeat of Bar 1)
                                (8, 1, 43),
                                (9, 0.5, 46),
                                (10, 0.5, 43),
                                (10.5, 0.5, 41),
                                (11, 1, 39),
                                # Bar 4 (variation)
                                (12, 0.5, 43),  # Short root note
                                (12.5, 0.5, 46),  # Syncopated note
                                (13, 0.5, 43),  # Short note
                                (13.5, 0.5, 41),  # Short note
                                (14, 0.5, 39),  # Short note
                                (14.5, 0.5, 41),  # Short note
                                (15, 0.5, 43),  # Short note
                                (15.5, 0.5, 46),  # End on a higher note
                            ]
                            
                            # Add bass notes according to our riff
                            for onset, duration, pitch in bass_riff:
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(int(onset))},
                                        "onset/tick": {str(int((onset % 1) * 24))},
                                        "offset/beat": {str(int(onset + duration))},
                                        "offset/tick": {str(int(((onset + duration) % 1) * 24))},
                                        "pitch": {str(pitch)},
                                    })
                                    .intersect(ec().velocity_constraint(80))  # Set a moderate velocity
                                    .force_active()
                                )
                            
                            # Set tempo to 96 (funky tempo)
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            
                            # Set tag to funk
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e