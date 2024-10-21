# Write a bass riff that goes with this drum beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a funky bass riff that complements the drum beat.
                            The bass riff will have a mix of short and long notes, with some syncopation.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define bass pitches for a funky riff (in E minor pentatonic scale)
                            bass_pitches = [40, 43, 45, 47, 48, 50, 52]  # E, G, A, B, C, D, E
                            
                            # Create the bass riff
                            bass_riff = [
                                # Bar 1
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"0"}, "offset/beat": {"0"}, "offset/tick": {"12"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[2])}, "onset/beat": {"1"}, "onset/tick": {"12"}, "offset/beat": {"2"}, "offset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[1])}, "onset/beat": {"2"}, "onset/tick": {"0"}, "offset/beat": {"2"}, "offset/tick": {"12"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[3])}, "onset/beat": {"3"}, "onset/tick": {"0"}, "offset/beat": {"3"}, "offset/tick": {"12"}}).force_active(),
                                
                                # Bar 2
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"4"}, "offset/beat": {"4"}, "offset/tick": {"12"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[4])}, "onset/beat": {"5"}, "onset/tick": {"0"}, "offset/beat": {"5"}, "offset/tick": {"12"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[2])}, "onset/beat": {"6"}, "onset/tick": {"0"}, "offset/beat": {"7"}, "offset/tick": {"0"}}).force_active(),
                                
                                # Bar 3
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"8"}, "offset/beat": {"8"}, "offset/tick": {"12"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[5])}, "onset/beat": {"9"}, "onset/tick": {"12"}, "offset/beat": {"10"}, "offset/tick": {"0"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[1])}, "onset/beat": {"10"}, "onset/tick": {"0"}, "offset/beat": {"10"}, "offset/tick": {"12"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[3])}, "onset/beat": {"11"}, "onset/tick": {"0"}, "offset/beat": {"11"}, "offset/tick": {"12"}}).force_active(),
                                
                                # Bar 4 (with some space for the drum triplets)
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"12"}, "offset/beat": {"12"}, "offset/tick": {"12"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[6])}, "onset/beat": {"13"}, "onset/tick": {"0"}, "offset/beat": {"13"}, "offset/tick": {"12"}}).force_active(),
                                ec().intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[4])}, "onset/beat": {"14"}, "onset/tick": {"0"}, "offset/beat": {"15"}, "offset/tick": {"0"}}).force_active(),
                            ]
                            
                            # Add the bass riff to the existing events
                            e += bass_riff
                            
                            # Set velocity for bass notes (slightly varied for groove)
                            for ev in e:
                                if ev.a["instrument"] == {"Bass"}:
                                    ev.intersect(ec().velocity_constraint(80 + (hash(str(ev.a["onset/beat"])) % 20)))
                            
                            # Set tempo to match the existing tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to funk
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e