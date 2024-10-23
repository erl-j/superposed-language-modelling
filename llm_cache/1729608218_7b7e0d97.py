# write a emotional trance piano lead
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create an emotional trance piano lead. This will involve creating a melodic piano part
                            that fits well with trance music, emphasizing emotion and atmosphere.
                            '''
                            # Remove any existing piano notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Create new piano events
                            piano_events = []
                            
                            # Add a series of longer, sustained notes for an emotional feel
                            for bar in range(4):
                                piano_events += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(bar * 4)},
                                        "offset/beat": {str(bar * 4 + 2)},  # Hold for 2 beats
                                        "pitch": {str(pitch) for pitch in range(60, 85)},  # Mid to high range for lead
                                    })
                                    .force_active()
                                ]
                            
                            # Add some shorter notes for melodic variation
                            for _ in range(8):
                                piano_events += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "pitch": {str(pitch) for pitch in range(60, 85)},
                                    })
                                    .force_active()
                                ]
                            
                            # Add some optional notes for the model to potentially use
                            piano_events += [
                                ec().intersect({"instrument": {"Piano"}}) for _ in range(10)
                            ]
                            
                            # Set velocity for emotional dynamics
                            piano_events = [ev.intersect(ec().velocity_constraint(80)) for ev in piano_events]
                            
                            # Combine with existing events
                            e += piano_events
                            
                            # Set tempo (assuming trance tempo, if not already set)
                            e = [ev.intersect(ec().tempo_constraint(138)) for ev in e]
                            
                            # Set tags
                            e = [ev.intersect({"tag": {"trance", "emotional", "piano", "lead"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e