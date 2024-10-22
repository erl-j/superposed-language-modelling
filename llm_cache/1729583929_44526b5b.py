# Start with some piano chords.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a constraint for piano chords to start a new beat.
                            '''
                            e = []
                            # Add piano chords
                            for bar in range(4):
                                for beat in [0, 2]:  # Add chords on the 1 and 3 of each bar
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "onset/beat": {str(beat + bar * 4)},
                                            "offset/beat": {str(beat + bar * 4 + 2)},  # Make each chord last for 2 beats
                                            "pitch": {str(pitch) for pitch in range(60, 85)},  # Middle to high range for chords
                                        })
                                        .force_active()
                                        for _ in range(3)  # 3 notes per chord
                                    ]
                            
                            # Add some optional piano notes for variation
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Piano"},
                                    "pitch": {str(pitch) for pitch in range(48, 85)},  # Wider range for optional notes
                                })
                                for _ in range(10)
                            ]
                            
                            # Set velocity to medium-high for clear, pronounced chords
                            e = [ev.intersect(ec().velocity_constraint(80)) for ev in e]
                            
                            # Set tempo (assuming a moderate tempo if not specified)
                            e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]
                            
                            # Set tag (assuming a general tag if not specified)
                            e = [ev.intersect({"tag": {"chords", "piano"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e