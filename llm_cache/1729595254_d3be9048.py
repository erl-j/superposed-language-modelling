# Add some toms to this beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add some toms to the existing beat while preserving the current structure.
                            Toms will be added as additional elements to create more depth and variation.
                            '''
                            # Keep all existing active events
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Add low, mid, and high toms
                            tom_pitches = {"41 (Drums)", "45 (Drums)", "48 (Drums)"}
                            
                            # Add 8 tom hits
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": tom_pitches})
                                    .force_active()
                                ]
                            
                            # Add some tom fills in the last bar
                            for beat in range(12, 16):  # Last bar
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Drums"},
                                        "pitch": tom_pitches,
                                        "onset/beat": {str(beat)},
                                    })
                                    .force_active()
                                ]
                            
                            # Add up to 5 optional tom hits
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": tom_pitches})
                                for _ in range(5)
                            ]
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to maintain the total number of events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e