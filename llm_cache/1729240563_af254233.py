# trap beat with rattling hats and 808s (35 midi pitch)
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a trap beat with rattling hats and 808s, we'll focus on creating a pattern with heavy 808 bass, snares, and rapid hi-hat patterns.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add 808 bass (pitch 35)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"35 (Drums)"}, "onset/beat": {str(bar * 4)}})
                                    .force_active()
                                ]
                                # Add some off-beat 808s for variation
                                e += [
                                    ec()
                                    .intersect({"pitch": {"35 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}})
                                    .force_active()
                                ]
            
                            # Add snares on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
            
                            # Add rattling hi-hats (closed and open)
                            # We'll add a dense pattern of hi-hats, some with lower velocity for the rattling effect
                            for beat in range(16):
                                # Closed hi-hats
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}})
                                    .intersect(ec().velocity_constraint(127))  # Full velocity
                                    .force_active()
                                ]
                                # Add 2 more closed hi-hats per beat with varying velocities
                                for tick in [8, 16]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {str(tick)}})
                                        .intersect(ec().velocity_constraint(70))  # Lower velocity for rattling effect
                                        .force_active()
                                    ]
            
                            # Add some open hi-hats for variation
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"46 (Drums)"}, "onset/beat": {str(bar * 4 + 3)}})
                                    .force_active()
                                ]
            
                            # Add up to 20 optional drum notes for additional complexity
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
            
                            # Set tempo to 140 BPM (typical for trap)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
            
                            # Set tag to 'electronic' and 'hip-hop' (closest to trap in the given list)
                            e = [ev.intersect({"tag": {"electronic", "hip-hop", "-"}}) for ev in e]
            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e  # return the events