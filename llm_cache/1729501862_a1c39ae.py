# Add some more hi-hats and toms.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a funkier beat with more hi-hats and toms, we'll add more hi-hat variations and some tom fills.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Add more hi-hats
                            # Closed hi-hats (42)
                            e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(16)]
                            
                            # Open hi-hats (46)
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add some hi-hat pedal (44) for variation
                            e += [ec().intersect({"pitch": {"44 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add toms
                            # High tom (50)
                            e += [ec().intersect({"pitch": {"50 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Mid tom (47)
                            e += [ec().intersect({"pitch": {"47 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Low tom (45)
                            e += [ec().intersect({"pitch": {"45 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add some tom fills in the last bar
                            e += [
                                ec()
                                .intersect({"pitch": {"45 (Drums)", "47 (Drums)", "50 (Drums)"}, "onset/beat": {"12", "13", "14", "15"}})
                                .force_active()
                                for _ in range(6)
                            ]
                            
                            # Ensure funk tag
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Set tempo to 96 (typical funk tempo)
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e