# A hard drill beat. One of the most distinctive features of drill drums is the hi-hat pattern, based on the tresillo rhythm. This rhythm, made up of two dotted eighth notes followed by an eighth note, creates a staggered, triplet feel.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a hard drill beat with the distinctive tresillo rhythm in the hi-hats.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set a faster tempo typical for drill beats
                            tempo = 140
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Kick pattern: typically on the 1 and in between 2 and 3
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}}).force_active(),
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}, "onset/tick": {"12"}}).force_active()
                                ]
                            
                            # Snare on 2 and 4
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Hi-hat tresillo pattern (3-3-2 rhythm)
                            # This creates the pattern for one bar, then repeats it for all 4 bars
                            tresillo_pattern = [
                                (0, 0), (1, 0), (2, 0),  # First dotted eighth
                                (3, 0), (4, 0), (5, 0),  # Second dotted eighth
                                (6, 0), (7, 0)           # Last eighth
                            ]
                            for bar in range(4):
                                for tick, _ in tresillo_pattern:
                                    e += [
                                        ec()
                                        .intersect({
                                            "pitch": {"42 (Drums)"},  # Closed hi-hat
                                            "onset/beat": {str(bar * 4 + tick // 3)},
                                            "onset/tick": {str(8 * (tick % 3))}
                                        })
                                        .force_active()
                                    ]
                            
                            # Add some open hi-hats for variation
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add some tom hits for additional rhythm
                            e += [ec().intersect({"pitch": {"45 (Drums)", "47 (Drums)", "48 (Drums)"}}).force_active() for _ in range(6)]
                            
                            # Add up to 20 optional drum notes for further complexity
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tag to electronic and driving
                            e = [ev.intersect({"tag": {"electronic", "driving"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e