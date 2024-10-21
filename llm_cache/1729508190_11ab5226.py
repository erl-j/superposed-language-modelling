# Write a bass riff that goes with this drum beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a funky bass riff that complements the drum beat, we'll create a pattern that emphasizes the groove
                            and syncopation typical of funk music. We'll use a combination of root notes, fifths, and some chromatic approaches.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define our bass pattern
                            bass_pattern = [
                                (0, 0, 35),   # Root on the 1
                                (0, 12, 35),  # Root on the "and" of 1
                                (1, 0, 42),   # Fifth on 2
                                (1, 12, 41),  # Chromatic approach to fifth
                                (2, 0, 42),   # Fifth on 3
                                (2, 12, 35),  # Root on the "and" of 3
                                (3, 0, 34),   # Chromatic approach to root
                                (3, 12, 35),  # Root on the "and" of 4
                            ]
                            
                            # Create bass notes for two bars (8 beats)
                            for bar in range(2):
                                for beat, tick, pitch in bass_pattern:
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/beat": {str(beat + bar * 4)},
                                            "onset/tick": {str(tick)},
                                            "pitch": {str(pitch)},
                                            "offset/beat": {str(beat + bar * 4 + 1)},  # Duration of 1 beat
                                            "offset/tick": {str(tick)}
                                        })
                                        .intersect(ec().velocity_constraint(100))  # Strong velocity for funk
                                        .force_active()
                                    )
                            
                            # For the last two bars, let's add some variation
                            variation_pattern = [
                                (8, 0, 35),   # Root on the 1
                                (8, 12, 37),  # Major third on the "and" of 1
                                (9, 0, 38),   # Fourth on 2
                                (9, 12, 40),  # Tritone on the "and" of 2
                                (10, 0, 42),  # Fifth on 3
                                (10, 12, 41), # Chromatic approach to fifth
                                (11, 0, 40),  # Fourth on 4
                                (11, 12, 38), # Minor third on the "and" of 4
                                (12, 0, 35),  # Root on 1 of last bar
                                (13, 0, 42),  # Fifth on 2 of last bar
                                (14, 0, 35),  # Root on 3 of last bar
                                (15, 0, 42),  # Fifth on 4 of last bar
                                (15, 16, 40), # Fourth on last triplet
                                (15, 20, 38), # Minor third on last triplet
                            ]
                            
                            for beat, tick, pitch in variation_pattern:
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(beat)},
                                        "onset/tick": {str(tick)},
                                        "pitch": {str(pitch)},
                                        "offset/beat": {str(beat + 1)},  # Duration of 1 beat
                                        "offset/tick": {str(tick)}
                                    })
                                    .intersect(ec().velocity_constraint(100))  # Strong velocity for funk
                                    .force_active()
                                )
                            
                            # Set funk tag
                            e = [ev.intersect({"tag": {"funk"}}) for ev in e]
                            
                            # Set tempo to 96 BPM (typical for funk)
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e