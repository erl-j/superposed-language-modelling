# add a bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a funky bass line that complements the existing drum pattern.
                            The bass line will have a mix of longer sustained notes and shorter, syncopated notes.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define bass pitches for a funky line (in MIDI pitch numbers)
                            bass_pitches = [36, 38, 40, 41, 43, 45, 46, 48]  # E2 to C3 range
                            
                            # Add a bass note on beat 1 of each bar
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(bass_pitches[0])},
                                        "onset/beat": {str(bar * 4)},
                                        "offset/beat": {str(bar * 4 + 2)}  # Hold for 2 beats
                                    })
                                    .intersect(ec().velocity_constraint(100))  # Strong velocity for emphasis
                                    .force_active()
                                ]
                            
                            # Add syncopated notes
                            syncopated_beats = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
                            for beat in syncopated_beats:
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in bass_pitches},
                                        "onset/beat": {str(beat)},
                                        "onset/tick": {"12"},  # Slightly off-beat
                                        "offset/beat": {str(beat + 1)},
                                        "offset/tick": {"0"}
                                    })
                                    .intersect(ec().velocity_constraint(80))  # Slightly softer
                                    .force_active()
                                ]
                            
                            # Add some ghost notes for extra funk
                            ghost_beats = [0, 2, 4, 6, 8, 10, 12, 14]
                            for beat in ghost_beats:
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in bass_pitches},
                                        "onset/beat": {str(beat)},
                                        "onset/tick": {"18"},  # Just before the next beat
                                        "offset/beat": {str(beat)},
                                        "offset/tick": {"23"}
                                    })
                                    .intersect(ec().velocity_constraint(40))  # Very soft for ghost notes
                                ]
                            
                            # Add some optional bass notes for variation
                            e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(10)]
                            
                            # Preserve tempo and tag
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e