# A simple techno beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a simple techno beat, we'll focus on a straightforward pattern with kick, snare, and hi-hat.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Kick drum on every quarter note
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Snare on the 2nd and 4th beat of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Hi-hat on every eighth note
                            for beat in range(32):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat // 2)}, "onset/tick": {str(12 * (beat % 2))}})
                                    .force_active()
                                ]
                            
                            # Add some optional percussion elements
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo to 128 BPM (typical for techno)
                            e = [ev.intersect(ec().tempo_constraint(128)) for ev in e]
                            
                            # Set tag to techno
                            e = [ev.intersect({"tag": {"techno", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e