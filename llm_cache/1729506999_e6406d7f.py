# Add a funky bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a funky bass line, we'll create a syncopated rhythm with some slides and ghost notes.
                            We'll use a mix of short and long notes, focusing on the root, fifth, and octave of a funky scale.
                            '''
                            # Remove existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define some funky bass pitches (in E minor pentatonic scale)
                            bass_pitches = [40, 43, 45, 47, 48, 52]  # E, G, A, B, C, E
                            
                            # Create a funky bassline
                            for bar in range(4):
                                # Root on the 1
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(bass_pitches[0])},
                                    "onset/beat": {str(bar * 4)},
                                    "offset/beat": {str(bar * 4 + 0.5)}
                                }).intersect(ec().velocity_constraint(100)).force_active())
                                
                                # Syncopated note on the "and" of 2
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(bass_pitches[2])},
                                    "onset/beat": {str(bar * 4 + 1)},
                                    "onset/tick": {"12"},
                                    "offset/beat": {str(bar * 4 + 2)}
                                }).intersect(ec().velocity_constraint(90)).force_active())
                                
                                # Ghost note on 3
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(bass_pitches[1])},
                                    "onset/beat": {str(bar * 4 + 2)},
                                    "offset/beat": {str(bar * 4 + 2.25)}
                                }).intersect(ec().velocity_constraint(60)).force_active())
                                
                                # Slide on 4
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(bass_pitches[3])},
                                    "onset/beat": {str(bar * 4 + 3)},
                                    "offset/beat": {str(bar * 4 + 3.5)}
                                }).intersect(ec().velocity_constraint(85)).force_active())
                                
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(bass_pitches[4])},
                                    "onset/beat": {str(bar * 4 + 3.5)},
                                    "offset/beat": {str(bar * 4 + 4)}
                                }).intersect(ec().velocity_constraint(85)).force_active())
                            
                            # Add some optional bass notes for variation
                            e += [ec().intersect({"instrument": {"Bass"}, "pitch": {str(p) for p in bass_pitches}}) for _ in range(8)]
                            
                            # Set tempo to 96 (typical for funk)
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            
                            # Set tag to funk
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e