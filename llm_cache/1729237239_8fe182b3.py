# A hard drill beat. One of the most distinctive features of drill drums is the hi-hat pattern, based on the tresillo rhythm. This rhythm, made up of two dotted eighth notes followed by an eighth note, creates a staggered, triplet feel. Make sure to include some 808 bass
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a hard drill beat with the distinctive tresillo hi-hat pattern and 808 bass.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add 808 bass (typically uses MIDI note 36)
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add kicks (typically on the 1 and in between beats)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}})
                                    .intersect({"onset/beat": {str(bar * 4)}})
                                    .force_active()
                                ]
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}})
                                    .intersect({"onset/beat": {str(bar * 4 + 2)}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
                            
                            # Add snares (typically on beats 2 and 4)
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add tresillo hi-hat pattern
                            for bar in range(4):
                                start_beat = bar * 4
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}})
                                    .intersect({"onset/beat": {str(start_beat)}, "onset/tick": {"0"}})
                                    .force_active(),
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}})
                                    .intersect({"onset/beat": {str(start_beat + 1)}, "onset/tick": {"12"}})
                                    .force_active(),
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}})
                                    .intersect({"onset/beat": {str(start_beat + 3)}, "onset/tick": {"0"}})
                                    .force_active()
                                ]
                            
                            # Add some open hi-hats for variation
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add some additional percussion elements (e.g., claps or additional snares)
                            e += [ec().intersect({"pitch": {"39 (Drums)", "40 (Drums)"}}).force_active() for _ in range(6)]
                            
                            # Set a typical drill tempo (usually between 130-150 BPM)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
                            
                            # Set tag to 'electronic' and 'urban' as drill is a subgenre of trap music
                            e = [ev.intersect({"tag": {"electronic", "urban", "-"}}) for ev in e]
                            
                            # Add optional drum notes for further complexity
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e