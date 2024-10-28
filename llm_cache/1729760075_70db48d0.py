# Now add a bassline. Simple with a fill at the end
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a simple bassline with a fill at the end.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            # Start over with bass notes
                            e = []
                            
                            # Simple bassline on beats 1 and 3 of each bar
                            for bar in range(3):  # First 3 bars
                                for beat in [0, 2]:  # Beats 1 and 3
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/beat": {str(beat + bar * 4)},
                                            "pitch": {str(pitch) for pitch in range(30, 46)},  # Low pitch range
                                            "offset/beat": {str(beat + bar * 4 + 1)}  # Hold for one beat
                                        })
                                        .force_active()
                                    ]
                            
                            # Fill in the last bar
                            for beat in range(12, 16):  # Last bar
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(beat)},
                                        "pitch": {str(pitch) for pitch in range(30, 50)},  # Slightly wider pitch range for fill
                                    })
                                    .force_active()
                                ]
                            
                            # Add up to 5 more optional bass notes for variation
                            e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(5)]
                            
                            # Intersect with current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e