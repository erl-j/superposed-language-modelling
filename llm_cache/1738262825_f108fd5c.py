# put down some piano chords
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add piano chords to the existing beat. We'll add a chord on each beat,
                            with some variation in duration and velocity.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside existing instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Start with an empty list for new events
                            e = []
                            
                            # Add piano chords on each beat
                            for bar in range(4):
                                for beat in range(4):
                                    chord_root = ec().intersect({
                                        "instrument": {"Piano"},
                                        "onset/global_tick": {str(beat * 24 + bar * 96)},
                                        "pitch": {str(pitch) for pitch in range(60, 85)},  # Middle range for root note
                                        "duration": {"1/4", "1/2"}  # Vary between quarter and half notes
                                    }).force_active()
                                    
                                    # Add two more notes to form a triad
                                    chord_third = chord_root.copy().intersect({"pitch": {str(pitch) for pitch in range(64, 89)}})
                                    chord_fifth = chord_root.copy().intersect({"pitch": {str(pitch) for pitch in range(67, 92)}})
                                    
                                    e += [chord_root, chord_third, chord_fifth]
            
                            # Add velocity variation
                            e = [ev.intersect(ec().velocity_constraint(80 + (i % 3) * 10)) for i, ev in enumerate(e)]
                            
                            # Preserve tempo from existing events
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve tag from existing events
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e