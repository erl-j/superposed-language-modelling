# give me a techno beat with open hats 4 on the floor kicks
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a techno beat with open hats and 4 on the floor kicks.
                            '''
                            e = []
                            # remove all drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # 4 on the floor kicks (one on each quarter note)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Open hi-hats on the offbeats (every other 8th note)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"46 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
                            
                            # Closed hi-hats on every 16th note
                            for beat in range(16):
                                for tick in [0, 6, 12, 18]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {str(tick)}})
                                        .force_active()
                                    ]
                            
                            # Snare on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add some optional percussion elements
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tempo to 128 BPM (typical for techno)
                            e = [ev.intersect(ec().tempo_constraint(128)) for ev in e]
                            
                            # Set tag to techno
                            e = [ev.intersect({"tag": {"techno", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e