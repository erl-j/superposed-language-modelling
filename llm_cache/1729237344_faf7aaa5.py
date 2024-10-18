# A hard drill beat with many toms. One of the most distinctive features of drill drums is the hi-hat pattern, based on the tresillo rhythm. This rhythm, made up of two dotted eighth notes followed by an eighth note, creates a staggered, triplet feel. Make sure to include some 808 bass (midi pitch 35)
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a hard drill beat with many toms, a distinctive hi-hat pattern based on the tresillo rhythm, and 808 bass.
                            '''
                            e = []
                            # remove all drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add 808 bass (pitch 35)
                            e += [ec().intersect({"pitch": {"35 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add kicks (pitch 36)
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add snares (pitch 38) on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add hi-hats (pitch 42) with tresillo rhythm
                            for bar in range(4):
                                base_beat = bar * 4
                                # First dotted eighth note
                                e += [ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(base_beat)}, "onset/tick": {"0"}}).force_active()]
                                # Second dotted eighth note
                                e += [ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(base_beat + 1)}, "onset/tick": {"12"}}).force_active()]
                                # Third eighth note
                                e += [ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(base_beat + 3)}, "onset/tick": {"0"}}).force_active()]
            
                            # Add many toms (pitches 45, 47, 48, 50)
                            tom_pitches = ["45 (Drums)", "47 (Drums)", "48 (Drums)", "50 (Drums)"]
                            for _ in range(20):
                                e += [ec().intersect({"pitch": set(tom_pitches)}).force_active()]
            
                            # Add some crash cymbals (pitch 49) for accent
                            e += [ec().intersect({"pitch": {"49 (Drums)"}}).force_active() for _ in range(4)]
            
                            # Add up to 20 optional drum notes for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
            
                            # Set tempo to 140 BPM (typical for drill beats)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
            
                            # Set tag to electronic (closest to drill in the given list)
                            e = [ev.intersect({"tag": {"electronic", "-"}}) for ev in e]
            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e  # return the events