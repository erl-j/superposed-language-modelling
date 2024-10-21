# bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a funky bass line that complements the drum beat.
                            The bass line will have a mix of sustained notes and short, punchy notes.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define bass pitches for a funky line (E, G, A, C)
                            bass_pitches = [40, 43, 45, 48]
                            
                            # Create a basic rhythm for the bass
                            for bar in range(4):
                                # Root note on the 1
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {str(bar*4)}, "offset/beat": {str(bar*4 + 2)}})
                                    .force_active()
                                )
                                
                                # Short notes on the 2 and 4
                                for beat in [1, 3]:
                                    e.append(
                                        ec()
                                        .intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[1])}, "onset/beat": {str(bar*4 + beat)}, "offset/beat": {str(bar*4 + beat + 0.5)}})
                                        .force_active()
                                    )
                                
                                # Syncopated note between 2 and 3
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[2])}, "onset/beat": {str(bar*4 + 2.5)}, "offset/beat": {str(bar*4 + 3)}})
                                    .force_active()
                                )
                            
                            # Add some variation in the last bar
                            e.append(
                                ec()
                                .intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[3])}, "onset/beat": {"12"}, "offset/beat": {"13"}})
                                .force_active()
                            )
                            e.append(
                                ec()
                                .intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[2])}, "onset/beat": {"13"}, "offset/beat": {"13.5"}})
                                .force_active()
                            )
                            e.append(
                                ec()
                                .intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[1])}, "onset/beat": {"13.5"}, "offset/beat": {"14"}})
                                .force_active()
                            )
                            e.append(
                                ec()
                                .intersect({"instrument": {"Bass"}, "pitch": {str(bass_pitches[0])}, "onset/beat": {"14"}, "offset/beat": {"16"}})
                                .force_active()
                            )
                            
                            # Add some optional bass notes for more variation
                            e += [ec().intersect({"instrument": {"Bass"}, "pitch": {str(p) for p in bass_pitches}}) for _ in range(8)]
                            
                            # Set velocity for punchy bass sound
                            e = [ev.intersect(ec().velocity_constraint(100)) for ev in e]
                            
                            # Set tempo (assuming funk tempo around 110 BPM)
                            e = [ev.intersect(ec().tempo_constraint(110)) for ev in e]
                            
                            # Set tag to funk
                            e = [ev.intersect({"tag": {"funk"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e