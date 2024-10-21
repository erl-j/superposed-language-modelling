# Add a funky bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a funky bass line, we'll create a more complex pattern with syncopation,
                            slides, and a mix of short and long notes. We'll use the Electric Bass instrument.
                            '''
                            # Remove existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define bass pitches for a funky line (in E minor pentatonic scale)
                            bass_pitches = [40, 43, 45, 47, 48, 50, 52, 55]
                            
                            # Create a funky bass pattern
                            for bar in range(4):
                                # Root note on the 1
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(bass_pitches[0])},
                                    "onset/beat": {str(bar * 4)},
                                    "offset/beat": {str(bar * 4 + 1)},
                                }).force_active())
                                
                                # Syncopated note on the "and" of 2
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(bass_pitches[2])},
                                    "onset/beat": {str(bar * 4 + 1)},
                                    "onset/tick": {"12"},
                                    "offset/beat": {str(bar * 4 + 2)},
                                }).force_active())
                                
                                # Short note on 3
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(bass_pitches[4])},
                                    "onset/beat": {str(bar * 4 + 2)},
                                    "offset/beat": {str(bar * 4 + 2)},
                                    "offset/tick": {"12"},
                                }).force_active())
                                
                                # Slide on 4
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(bass_pitches[3])},
                                    "onset/beat": {str(bar * 4 + 3)},
                                    "offset/beat": {str(bar * 4 + 3)},
                                    "offset/tick": {"12"},
                                }).force_active())
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(bass_pitches[5])},
                                    "onset/beat": {str(bar * 4 + 3)},
                                    "onset/tick": {"12"},
                                    "offset/beat": {str(bar * 4 + 4)},
                                }).force_active())
                            
                            # Add some ghost notes for extra funkiness
                            for _ in range(8):
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(pitch) for pitch in bass_pitches},
                                }).intersect(ec().velocity_constraint(30)))
                            
                            # Set all bass notes to use Electric Bass sound
                            e = [ev.intersect({"instrument": {"Electric Bass (finger)"}}) if "Bass" in ev.a["instrument"] else ev for ev in e]
                            
                            # Ensure funk tag
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Set tempo to a funky 110 BPM
                            e = [ev.intersect(ec().tempo_constraint(110)) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e