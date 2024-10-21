# Add some more hi-hats
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a funkier beat with more hi-hats, we'll add more closed and open hi-hats,
                            and create a more complex hi-hat pattern.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove existing hi-hats
                            e = [ev for ev in e if ev.a["pitch"].isdisjoint({"42 (Drums)", "46 (Drums)"})]
                            
                            # Add 16 closed hi-hats (one for each 16th note)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}})
                                    .intersect({"onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Add 8 open hi-hats on off-beats
                            for beat in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"46 (Drums)"}})
                                    .intersect({"onset/beat": {str(int(beat))}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
                            
                            # Add some hi-hat variations
                            # 1. Add some ghost notes (lower velocity)
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}})
                                    .intersect(ec().velocity_constraint(30))
                                    .force_active()
                                ]
                            
                            # 2. Add some hi-hat chokes (quickly closed open hi-hats)
                            for _ in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"46 (Drums)"}})
                                    .intersect(ec().duration_constraint(0.25))
                                    .force_active()
                                ]
                            
                            # 3. Add some hi-hat foot splashes
                            for _ in range(2):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"44 (Drums)"}})
                                    .force_active()
                                ]
                            
                            # Maintain the funk feel
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Important: always pad with empty notes!
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events