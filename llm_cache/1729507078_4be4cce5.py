# Add a bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a funky bass line, we'll create a pattern that complements the drum beat,
                            focusing on the root notes and some rhythmic variations.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define bass pattern
                            bass_pattern = [
                                (0, 0),   # On the 1
                                (0, 12),  # Offbeat
                                (1, 0),   # On the 2
                                (1, 12),  # Offbeat
                                (2, 0),   # On the 3
                                (2, 12),  # Offbeat
                                (3, 0),   # On the 4
                                (3, 8),   # Sixteenth note before 1
                                (3, 16)   # Eighth note before 1
                            ]
                            
                            # Add bass notes for each bar
                            for bar in range(4):
                                for beat, tick in bass_pattern:
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/beat": {str(bar * 4 + beat)},
                                            "onset/tick": {str(tick)},
                                            "pitch": {str(pitch) for pitch in range(35, 55)},  # Bass range
                                            "offset/beat": {str(bar * 4 + beat + 1)},  # Duration of 1 beat
                                            "offset/tick": {str(tick)}
                                        })
                                        .intersect(ec().velocity_constraint(80))  # Medium-high velocity for funk
                                        .force_active()
                                    )
                            
                            # Add some variation in the last bar
                            for tick in [0, 6, 12, 18]:
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {"15"},  # Last beat of the last bar
                                        "onset/tick": {str(tick)},
                                        "pitch": {str(pitch) for pitch in range(40, 60)},  # Slightly higher range for variation
                                        "offset/beat": {"16"},
                                        "offset/tick": {str((tick + 6) % 24)}  # Short duration for funky effect
                                    })
                                    .intersect(ec().velocity_constraint(90))  # Higher velocity for emphasis
                                    .force_active()
                                )
                            
                            # Add some ghost notes for additional funk
                            for _ in range(5):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in range(35, 55)},
                                    })
                                    .intersect(ec().velocity_constraint(40))  # Low velocity for ghost notes
                                )
                            
                            # Ensure funk tag
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Set tempo to a funky 96 BPM if not already set
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e