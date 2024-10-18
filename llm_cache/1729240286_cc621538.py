# uptempo jungle beat. Have some 808s (midi pitch 35)
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a fast-paced jungle beat with 808s, we'll use a combination of breakbeat-style drums and 808 bass hits.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add kick drum (36) - typical breakbeat pattern
                            kick_pattern = [0, 2, 4, 6, 10]
                            for beat in kick_pattern:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}})
                                    .intersect({"onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Add snare drum (38) - on 2 and 4, plus some syncopation
                            snare_pattern = [4, 7, 12, 14]
                            for beat in snare_pattern:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}})
                                    .intersect({"onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Add closed hi-hats (42) - sixteenth note pattern
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}})
                                    .intersect({"onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Add open hi-hats (46) - occasional accents
                            open_hat_pattern = [2, 6, 10, 14]
                            for beat in open_hat_pattern:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"46 (Drums)"}})
                                    .intersect({"onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Add 808 bass drum (35) - syncopated pattern
                            bass_pattern = [1, 3, 5, 9, 11, 13]
                            for beat in bass_pattern:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"35 (Drums)"}})
                                    .intersect({"onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Add some ghost notes and additional percussion
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set a fast tempo for jungle (170-180 BPM)
                            tempo = 175
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tags
                            e = [ev.intersect({"tag": {"jungle", "electronic", "breakbeat", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e