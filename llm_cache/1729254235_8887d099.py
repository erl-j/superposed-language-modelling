# Give me a classic reggaeton beat over 4 bars at 120 bpm. make sure to include the dembow rhythm. 
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a classic reggaeton beat with the dembow rhythm over 4 bars at 120 bpm.
                            '''
                            e = []
                            # remove all drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Dembow rhythm pattern (repeated for each bar)
                            for bar in range(4):
                                # Kick drum (36) on beats 1 and 3
                                for beat in [0, 2]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                                
                                # Snare drum (38) on beats 2 and 4
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                                
                                # Dembow rhythm on rim shot (37)
                                e += [
                                    ec()
                                    .intersect({"pitch": {"37 (Drums)"}, "onset/beat": {str(bar * 4)}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
                                e += [
                                    ec()
                                    .intersect({"pitch": {"37 (Drums)"}, "onset/beat": {str(1 + bar * 4)}, "onset/tick": {"0"}})
                                    .force_active()
                                ]
                                e += [
                                    ec()
                                    .intersect({"pitch": {"37 (Drums)"}, "onset/beat": {str(2 + bar * 4)}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
                                e += [
                                    ec()
                                    .intersect({"pitch": {"37 (Drums)"}, "onset/beat": {str(3 + bar * 4)}, "onset/tick": {"0"}})
                                    .force_active()
                                ]
            
                            # Hi-hat pattern (42 for closed, 46 for open)
                            for bar in range(4):
                                for beat in range(4):
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                                    # Open hi-hat on the off-beats
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"46 (Drums)"}, "onset/beat": {str(beat + bar * 4)}, "onset/tick": {"12"}})
                                        .force_active()
                                    ]
            
                            # Add some optional percussion hits
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
            
                            # Set tempo to 120 bpm
                            e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]
            
                            # Set tag to latin (closest to reggaeton in the given list)
                            e = [ev.intersect({"tag": {"latin", "-"}}) for ev in e]
            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e