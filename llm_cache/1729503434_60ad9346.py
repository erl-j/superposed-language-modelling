# Add a melody
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a melody that complements the existing funk beat and bassline, we'll add a lead instrument (e.g., electric piano or synth) with a funky, syncopated rhythm.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            # Remove any existing melody notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Electric Piano", "Synth Lead"})]
                            
                            # Define melody instrument
                            melody_instrument = "Electric Piano"
                            
                            # Add main melody notes
                            for bar in range(4):
                                # Add a note on the 1 of each bar
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {melody_instrument},
                                        "onset/beat": {str(bar * 4)},
                                        "offset/beat": {str(bar * 4 + 1)},  # Hold for one beat
                                    })
                                    .force_active()
                                ]
                                
                                # Add syncopated notes
                                syncopated_beats = [bar * 4 + 1.5, bar * 4 + 2.75, bar * 4 + 3.5]
                                for beat in syncopated_beats:
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {melody_instrument},
                                            "onset/beat": {str(int(beat))},
                                            "onset/tick": {str(int((beat % 1) * 24))},
                                            "offset/beat": {str(int(beat + 0.5))},
                                            "offset/tick": {str(int(((beat + 0.5) % 1) * 24))},
                                        })
                                        .force_active()
                                    ]
            
                            # Add some shorter notes for rhythmic variety
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {melody_instrument},
                                        "offset/beat": ec().offset_constraint(lambda onset: onset + 0.25),  # Short 16th note
                                    })
                                    .force_active()
                                ]
            
                            # Add some longer notes for contrast
                            for _ in range(4):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {melody_instrument},
                                        "offset/beat": ec().offset_constraint(lambda onset: onset + 2),  # Hold for two beats
                                    })
                                    .force_active()
                                ]
            
                            # Ensure melody stays within a reasonable pitch range
                            e = [ev.intersect({"pitch": pitch_range[36:72]}) for ev in e]  # C3 to C6
            
                            # Add some optional melody notes
                            e += [ec().intersect({"instrument": {melody_instrument}}) for _ in range(10)]
            
                            # Maintain funk feel
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e