# A hard drill beat. One of the most distinctive features of drill drums is the hi-hat pattern, based on the tresillo rhythm. This rhythm, made up of two dotted eighth notes followed by an eighth note, creates a staggered, triplet feel. Make sure to include some 808 bass (midi pitch 35)
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a hard drill beat with the distinctive tresillo hi-hat pattern and 808 bass.
                            '''
                            e = []
                            # Remove all existing drums
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
                            
                            # Add tresillo hi-hat pattern (pitch 42)
                            # The pattern repeats every 3 beats, so we'll add it for each bar
                            for bar in range(4):
                                start_beat = bar * 4
                                e += [
                                    ec().intersect({"pitch": {"42 (Drums)"}})
                                    .intersect({"onset/beat": {str(start_beat)}, "onset/tick": {"0"}})
                                    .force_active(),
                                    ec().intersect({"pitch": {"42 (Drums)"}})
                                    .intersect({"onset/beat": {str(start_beat + 1)}, "onset/tick": {"12"}})
                                    .force_active(),
                                    ec().intersect({"pitch": {"42 (Drums)"}})
                                    .intersect({"onset/beat": {str(start_beat + 2)}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
                            
                            # Add some open hi-hats (pitch 46) for variation
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add some additional percussion hits (e.g., claps on pitch 39)
                            e += [ec().intersect({"pitch": {"39 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add up to 20 optional drum notes for further complexity
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tempo to 140 BPM (typical for drill beats)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
                            
                            # Set tag to 'electronic' and 'urban' as drill is a subgenre of trap music
                            e = [ev.intersect({"tag": {"electronic", "urban", "-"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events