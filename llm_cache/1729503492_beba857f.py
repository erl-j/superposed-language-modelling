# Add a piano melody
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a piano melody, we'll add a series of piano notes with some constraints to create a melodic line.
                            '''
                            # Remove inactive notes and any existing piano notes
                            e = [ev for ev in e if ev.is_active() and ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Define piano melody parameters
                            num_piano_notes = 16  # Number of piano notes to add
                            piano_pitch_range = range(60, 84)  # C4 to C6
                            
                            # Add piano notes
                            for i in range(num_piano_notes):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "pitch": set(map(str, piano_pitch_range)),
                                        "onset/beat": set(map(str, range(16))),  # Allow notes on any beat
                                        "onset/tick": set(map(str, range(24))),  # Allow any tick within a beat
                                        "velocity": set(map(str, range(70, 110))),  # Medium to loud velocity
                                    })
                                    .force_active()
                                ]
                            
                            # Add some longer notes for variety
                            for _ in range(4):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "pitch": set(map(str, piano_pitch_range)),
                                        "onset/beat": set(map(str, range(16))),
                                        "onset/tick": set(map(str, range(24))),
                                        "offset/beat": set(map(str, range(1, 17))),  # Ensure note lasts at least 1 beat
                                        "velocity": set(map(str, range(70, 110))),
                                    })
                                    .force_active()
                                ]
                            
                            # Ensure some rhythmic variety
                            syncopated_beats = [0, 2, 6, 10, 14]  # Some syncopated beat positions
                            for beat in syncopated_beats:
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "pitch": set(map(str, piano_pitch_range)),
                                        "onset/beat": {str(beat)},
                                        "onset/tick": {"12"},  # Place on the "and" of the beat
                                        "velocity": set(map(str, range(80, 120))),  # Slightly accented
                                    })
                                    .force_active()
                                ]
                            
                            # Ensure the melody starts and ends on strong beats
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Piano"},
                                    "pitch": set(map(str, piano_pitch_range)),
                                    "onset/beat": {"0"},  # Start on the first beat
                                    "onset/tick": {"0"},
                                    "velocity": set(map(str, range(90, 127))),  # Strong velocity
                                })
                                .force_active()
                            ]
                            
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Piano"},
                                    "pitch": set(map(str, piano_pitch_range)),
                                    "onset/beat": {"15"},  # End on the last beat
                                    "onset/tick": {"0"},
                                    "velocity": set(map(str, range(90, 127))),  # Strong velocity
                                })
                                .force_active()
                            ]
                            
                            # Maintain the funk tag
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e