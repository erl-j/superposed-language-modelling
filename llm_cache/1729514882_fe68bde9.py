# Write a bass riff that goes with this drum beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a funky bass riff that complements the existing drum beat.
                            The bass riff will have a mix of quarter notes, eighth notes, and some syncopation.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define our bass riff
                            bass_riff = [
                                # Bar 1
                                (0, 0, 1, 0),  # Root note on beat 1
                                (1, 12, 2, 0),  # Syncopated note on the "and" of 2
                                (2, 0, 2, 12),  # Quarter note on beat 3
                                (3, 0, 3, 12),  # Quarter note on beat 4
                                # Bar 2
                                (4, 0, 4, 12),  # Quarter note on beat 1
                                (5, 0, 5, 12),  # Quarter note on beat 2
                                (6, 0, 6, 12),  # Quarter note on beat 3
                                (7, 0, 7, 12),  # Quarter note on beat 4
                                # Bar 3
                                (8, 0, 8, 12),   # Quarter note on beat 1
                                (9, 0, 9, 12),   # Quarter note on beat 2
                                (10, 0, 10, 12), # Quarter note on beat 3
                                (11, 0, 11, 12), # Quarter note on beat 4
                                # Bar 4 (with some syncopation)
                                (12, 0, 12, 12),  # Quarter note on beat 1
                                (13, 12, 14, 0),  # Syncopated note on the "and" of 2
                                (14, 12, 15, 0),  # Syncopated note on the "and" of 3
                                (15, 12, 16, 0),  # Syncopated note on the "and" of 4
                            ]
                            
                            # Add bass notes according to our riff
                            for onset_beat, onset_tick, offset_beat, offset_tick in bass_riff:
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(onset_beat)},
                                        "onset/tick": {str(onset_tick)},
                                        "offset/beat": {str(offset_beat)},
                                        "offset/tick": {str(offset_tick)},
                                        "pitch": {str(pitch) for pitch in range(35, 55)},  # Range for funky bass
                                    })
                                    .intersect(ec().velocity_constraint(80))  # Medium-high velocity for funk
                                    .force_active()
                                )
                            
                            # Set tempo to match the funk beat (assuming 96 BPM from previous constraint)
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            
                            # Set tag to funk
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Add some optional bass notes for variation
                            e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(5)]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e