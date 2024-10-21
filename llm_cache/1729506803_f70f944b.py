# Add a funky bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a funky bass line, we'll create a syncopated rhythm with a mix of short and long notes,
                            focusing on the root, fifth, and octave of a funk-friendly scale (e.g., E minor pentatonic).
                            We'll also ensure it interacts well with the existing drum pattern.
                            '''
                            # Remove existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define funky bass pitches (E minor pentatonic: E, G, A, B, D)
                            bass_pitches = {40, 43, 45, 47, 50, 52, 55, 57, 59, 62}  # Two octaves
                            
                            # Create syncopated rhythm for bass
                            bass_rhythms = [
                                (0, 0), (0, 12), (1, 0), (1, 12), (2, 8), (3, 0),  # Bar 1
                                (4, 4), (4, 16), (5, 8), (6, 0), (7, 0),           # Bar 2
                                (8, 0), (8, 12), (9, 8), (10, 0), (10, 12),        # Bar 3
                                (12, 0), (12, 12), (13, 8), (14, 0), (15, 0)       # Bar 4
                            ]
                            
                            # Add bass notes
                            for beat, tick in bass_rhythms:
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Bass"}})
                                    .intersect({"pitch": bass_pitches})
                                    .intersect({"onset/beat": {str(beat)}, "onset/tick": {str(tick)}})
                                    .intersect(ec().velocity_constraint(80))  # Moderately strong velocity
                                    .force_active()
                                )
                            
                            # Add some longer sustained notes
                            sustained_notes = [(2, 0), (6, 0), (10, 0), (14, 0)]
                            for beat, tick in sustained_notes:
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Bass"}})
                                    .intersect({"pitch": bass_pitches})
                                    .intersect({"onset/beat": {str(beat)}, "onset/tick": {str(tick)}})
                                    .intersect({"offset/beat": {str(beat + 1)}, "offset/tick": {str(tick)}})
                                    .intersect(ec().velocity_constraint(90))  # Stronger velocity for emphasis
                                    .force_active()
                                )
                            
                            # Add some ghost notes (quieter, shorter notes) for extra funkiness
                            ghost_notes = [(1, 18), (5, 18), (9, 18), (13, 18)]
                            for beat, tick in ghost_notes:
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Bass"}})
                                    .intersect({"pitch": bass_pitches})
                                    .intersect({"onset/beat": {str(beat)}, "onset/tick": {str(tick)}})
                                    .intersect(ec().velocity_constraint(50))  # Lower velocity for ghost notes
                                    .force_active()
                                )
                            
                            # Set funk tag
                            e = [ev.intersect({"tag": {"funk"}}) for ev in e]
                            
                            # Set tempo (assuming 96 BPM for a funky groove)
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e