# A dense beat, hard drill beat with many toms. One of the most distinctive features of drill drums is the hi-hat pattern, based on the tresillo rhythm. This rhythm, made up of two dotted eighth notes followed by an eighth note, creates a staggered, triplet feel. Make sure to include some 808 bass (midi pitch 35)
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a dense, hard drill beat with many toms, tresillo hi-hat pattern, and 808 bass.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add 808 bass (pitch 35)
                            e += [ec().intersect({"pitch": {"35 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add kicks (pitch 36)
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(12)]
                            
                            # Add snares (pitch 38)
                            e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add toms (pitches 45, 47, 48, 50)
                            tom_pitches = ["45 (Drums)", "47 (Drums)", "48 (Drums)", "50 (Drums)"]
                            for _ in range(20):
                                e += [ec().intersect({"pitch": {tom_pitch}}).force_active() for tom_pitch in tom_pitches]
                            
                            # Add tresillo hi-hat pattern (pitch 42)
                            for bar in range(4):
                                for beat in range(4):
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active(),
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}, "onset/tick": {"12"}})
                                        .force_active(),
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}, "onset/tick": {"18"}})
                                        .force_active()
                                    ]
                            
                            # Add some open hi-hats (pitch 46)
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(6)]
                            
                            # Add some crash cymbals (pitch 49)
                            e += [ec().intersect({"pitch": {"49 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add up to 20 optional drum notes for extra density
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tempo to 140 BPM (typical for drill)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
                            
                            # Set tag to 'electronic' and 'groove' as drill is not a specific option
                            e = [ev.intersect({"tag": {"electronic", "groove"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events