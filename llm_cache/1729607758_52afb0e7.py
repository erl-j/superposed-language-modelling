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
                            
                            # Create a main melodic line
                            for beat in range(0, 16, 2):  # Create notes every two beats for a flowing melody
                                piano_events.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(beat)},
                                        "pitch": {str(pitch) for pitch in range(60, 84)},  # Mid to high range for lead
                                    })
                                    .intersect(ec().velocity_constraint(80))  # Moderately loud for emphasis
                                    .force_active()
                                )
                            
                            # Add some shorter notes for variation
                            for _ in range(8):
                                piano_events.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "pitch": {str(pitch) for pitch in range(72, 96)},  # Higher range for accents
                                    })
                                    .intersect(ec().velocity_constraint(70))
                                    .force_active()
                                )
                            
                            # Add some longer, sustained notes for emotion
                            for beat in [0, 8]:  # Two long notes, one at the start and one halfway
                                piano_events.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(beat)},
                                        "offset/beat": {str(beat + 4)},  # 4 beats long
                                        "pitch": {str(pitch) for pitch in range(48, 72)},  # Lower range for pads
                                    })
                                    .intersect(ec().velocity_constraint(60))  # Softer for atmosphere
                                    .force_active()
                                )
                            
                            # Add the piano events to the existing events
                            e += piano_events
                            
                            # Set the tag to trance and emotional
                            e = [ev.intersect({"tag": {"trance", "emotional"}}) for ev in e]
                            
                            # Preserve the existing tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Pad with inactive events to reach n_events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e