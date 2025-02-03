# write a drum beat with many kicks
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a drum beat with many kicks, complemented by other drum elements.
                            '''
                            e = []
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add many kicks (20 in total)
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(20)]
                            
                            # Add some snares (8 in total)
                            e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add hi-hats (12 in total, mix of closed and open)
                            e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(8)]  # closed hi-hat
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]  # open hi-hat
                            
                            # Add some tom hits for variety (6 in total)
                            e += [ec().intersect({"pitch": {"45 (Drums)", "47 (Drums)", "50 (Drums)"}}).force_active() for _ in range(6)]
                            
                            # Add up to 10 optional drum hits for additional complexity
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo (assuming a moderate tempo for a kick-heavy beat)
                            e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]
                            
                            # Set tag (assuming this could fit under 'rock' or 'dance-eletric')
                            e = [ev.intersect({"tag": {"rock", "dance-eletric"}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e