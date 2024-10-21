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
                                    .intersect({"onset/tick": {"0", "6", "12", "18"}})
                                    .force_active()
                                ]
                            
                            # Add 8 open hi-hats on off-beats
                            for beat in range(16):
                                if beat % 2 != 0:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"46 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat)}})
                                        .intersect({"onset/tick": {"0"}})
                                        .force_active()
                                    ]
                            
                            # Add some syncopated hi-hats
                            syncopated_beats = [2, 6, 10, 14]
                            for beat in syncopated_beats:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}})
                                    .intersect({"onset/beat": {str(beat)}})
                                    .intersect({"onset/tick": {"18"}})
                                    .force_active()
                                ]
                            
                            # Add some hi-hat variations
                            variation_beats = [3, 7, 11, 15]
                            for beat in variation_beats:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)", "46 (Drums)"}})
                                    .intersect({"onset/beat": {str(beat)}})
                                    .intersect({"onset/tick": {"12"}})
                                    .force_active()
                                ]
                            
                            # Add up to 10 optional hi-hat notes for more variation
                            e += [
                                ec()
                                .intersect({"pitch": {"42 (Drums)", "46 (Drums)"}})
                                for _ in range(10)
                            ]
                            
                            # Ensure the funk tag is set
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e