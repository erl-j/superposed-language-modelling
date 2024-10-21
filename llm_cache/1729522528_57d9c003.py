# add a bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a funky bass line that complements the existing drum pattern.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define bass pitches for a funky line
                            bass_pitches = [36, 38, 40, 41, 43, 45, 46, 48]  # E2 to C3 range
                            
                            # Add a bass note on beat 1 of each bar
                            for bar in range(4):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(bar * 4)},
                                        "pitch": {str(bass_pitches[0])},
                                        "offset/beat": {str(bar * 4 + 1)}  # Hold for one beat
                                    })
                                    .intersect(ec().velocity_constraint(100))  # Strong velocity for emphasis
                                    .force_active()
                                )
                            
                            # Add syncopated bass notes
                            syncopated_beats = [1, 2.5, 3.5]  # Syncopated rhythm
                            for bar in range(4):
                                for beat in syncopated_beats:
                                    onset_beat = int(bar * 4 + beat)
                                    onset_tick = int((beat % 1) * 24)
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/beat": {str(onset_beat)},
                                            "onset/tick": {str(onset_tick)},
                                            "pitch": {str(pitch) for pitch in bass_pitches[1:]},  # Use other pitches
                                            "offset/beat": {str(onset_beat + 1 if onset_beat < 15 else 15)},
                                            "offset/tick": {str(onset_tick)}
                                        })
                                        .intersect(ec().velocity_constraint(80))  # Slightly lower velocity
                                        .force_active()
                                    )
                            
                            # Add some optional ghost notes for variation
                            for _ in range(8):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in bass_pitches}
                                    })
                                    .intersect(ec().velocity_constraint(60))  # Lower velocity for ghost notes
                                )
                            
                            # Set tempo and tag
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            e = [ev.intersect({"tag": {"funk"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e