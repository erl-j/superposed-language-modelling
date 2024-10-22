# Create an emotional piano chord progression
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create an emotional piano chord progression. This will involve:
                            1. Removing any existing piano notes
                            2. Adding piano chords on each beat
                            3. Setting appropriate pitch ranges for emotional chords
                            4. Adjusting velocity for dynamics
                            5. Preserving the existing tempo and adding an emotional tag
                            '''
                            # Remove existing piano notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Add piano chords on each beat
                            for beat in range(16):  # 4 bars * 4 beats
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(beat)},
                                        # Pitch range for emotional chords (middle to high register)
                                        "pitch": {str(pitch) for pitch in range(60, 84)},
                                    })
                                    .intersect(ec().velocity_constraint(80))  # Moderately loud for expression
                                    .force_active()
                                    for _ in range(3)  # 3 notes per chord
                                ]
            
                            # Add some variation with occasional higher notes
                            for _ in range(4):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "pitch": {str(pitch) for pitch in range(84, 96)},
                                    })
                                    .intersect(ec().velocity_constraint(70))  # Slightly softer for contrast
                                ]
            
                            # Preserve the existing tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
            
                            # Set tags to emotional and piano
                            e = [ev.intersect({"tag": {"emotional", "piano"}}) for ev in e]
            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e