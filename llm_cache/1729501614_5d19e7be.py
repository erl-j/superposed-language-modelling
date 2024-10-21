# Write a bassline than descends
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a descending bassline, we'll create a series of bass notes
                            that gradually decrease in pitch over the course of the loop.
                            '''
                            # Remove existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define a descending bass pattern
                            bass_pitches = [43, 41, 39, 38, 36, 34, 32, 31]  # G2, F2, D#2, D2, C2, A#1, G#1, G1
                            bass_rhythm = [0, 2, 4, 6, 8, 10, 12, 14]  # Every half note
                            
                            # Add descending bass notes
                            for i, (pitch, beat) in enumerate(zip(bass_pitches, bass_rhythm)):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch)},
                                        "onset/beat": {str(beat)},
                                        "duration/beat": {"1"},  # Quarter note duration
                                    })
                                    .force_active()
                                )
                            
                            # Add some optional bass notes for variation
                            for _ in range(4):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(p) for p in range(31, 44)},  # G1 to G2
                                    })
                                )
                            
                            # Set tempo to 96 (assuming funk tempo from previous prompt)
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            
                            # Set tag to funk
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e