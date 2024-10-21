# bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a funky bass line that complements the drum beat.
                            The bass line will have a mix of sustained notes and short, punchy notes.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define bass pitches for a funky line (in MIDI numbers)
                            bass_pitches = [36, 38, 40, 41, 43, 45, 46, 48]  # E2 to C3 range
                            
                            # Add a bass note on beat 1 of each bar
                            for bar in range(4):
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Bass"}, "onset/beat": {str(bar * 4)}, "pitch": {str(bass_pitches[0])}})
                                    .intersect(ec().offset_constraint(bar * 4 + 2, 0))  # Sustain for 2 beats
                                    .force_active()
                                )
                            
                            # Add syncopated notes
                            syncopated_beats = [0.5, 1.5, 2.5, 3.5]
                            for bar in range(4):
                                for beat in syncopated_beats:
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"}, 
                                            "onset/beat": {str(int(bar * 4 + beat))},
                                            "onset/tick": {"12"}, # Half-beat offset
                                            "pitch": {str(pitch) for pitch in bass_pitches[1:]}
                                        })
                                        .intersect(ec().offset_constraint(int(bar * 4 + beat), 18))  # Short, punchy notes
                                        .force_active()
                                    )
                            
                            # Add some ghost notes for extra funk
                            for _ in range(8):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"}, 
                                        "pitch": {str(pitch) for pitch in bass_pitches}
                                    })
                                    .intersect(ec().velocity_constraint(40))  # Lower velocity for ghost notes
                                )
                            
                            # Set tempo
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]  # Funky tempo around 96 BPM
                            
                            # Set tag
                            e = [ev.intersect({"tag": {"funk"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e