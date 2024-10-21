# Write a bass riff that goes with this drum beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a funky bass riff that complements the existing drum beat.
                            The bass riff will have a combination of root notes, fifths, and some rhythmic variations.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define some common funk bass pitches (E, G, A in different octaves)
                            bass_pitches = [40, 43, 45, 52, 55, 57]  # E2, G2, A2, E3, G3, A3
                            
                            # Create the bass riff
                            bass_riff = [
                                # Bar 1
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"0"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[3])}, "onset/beat": {"0"}, "onset/tick": {"12"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[1])}, "onset/beat": {"1"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[2])}, "onset/beat": {"2"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"3"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[3])}, "onset/beat": {"3"}, "onset/tick": {"12"}}).force_active(),
                                
                                # Bar 2
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[1])}, "onset/beat": {"4"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[4])}, "onset/beat": {"4"}, "onset/tick": {"12"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[2])}, "onset/beat": {"5"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"6"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[1])}, "onset/beat": {"7"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[4])}, "onset/beat": {"7"}, "onset/tick": {"12"}}).force_active(),
                                
                                # Bar 3 (repeat of Bar 1)
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"8"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[3])}, "onset/beat": {"8"}, "onset/tick": {"12"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[1])}, "onset/beat": {"9"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[2])}, "onset/beat": {"10"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"11"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[3])}, "onset/beat": {"11"}, "onset/tick": {"12"}}).force_active(),
                                
                                # Bar 4 (variation for the last bar with triplets)
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[1])}, "onset/beat": {"12"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[2])}, "onset/beat": {"12"}, "onset/tick": {"8"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"12"}, "onset/tick": {"16"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[3])}, "onset/beat": {"13"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[1])}, "onset/beat": {"13"}, "onset/tick": {"8"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[4])}, "onset/beat": {"13"}, "onset/tick": {"16"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[2])}, "onset/beat": {"14"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"14"}, "onset/tick": {"8"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[3])}, "onset/beat": {"14"}, "onset/tick": {"16"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[1])}, "onset/beat": {"15"}, "onset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[2])}, "onset/beat": {"15"}, "onset/tick": {"8"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"15"}, "onset/tick": {"16"}}).force_active(),
                            ]
                            
                            # Add the bass riff to the existing events
                            e += bass_riff
                            
                            # Set velocity for bass notes (slightly varied for groove)
                            for ev in e:
                                if ev.a["instrument"] == {"Bass"}:
                                    ev.intersect(ec().velocity_constraint(80 + (hash(str(ev.a)) % 20)))
                            
                            # Set funk tag
                            e = [ev.intersect({"tag": {"funk"}}) for ev in e]
                            
                            # Set tempo (assuming 96 BPM for funk)
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            
                            # Add some optional bass notes for variation
                            e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(5)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in