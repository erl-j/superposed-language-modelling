# create a drum beat with lots of percussion
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a drum beat with lots of percussion, including kicks, snares, hi-hats, and various percussion instruments.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add kick drum (bass drum)
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add snare drum
                            e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(6)]
                            
                            # Add closed hi-hat
                            e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(16)]
                            
                            # Add open hi-hat
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add tom-toms
                            e += [ec().intersect({"pitch": {"45 (Drums)", "47 (Drums)", "48 (Drums)"}}).force_active() for _ in range(6)]
                            
                            # Add crash cymbal
                            e += [ec().intersect({"pitch": {"49 (Drums)"}}).force_active() for _ in range(2)]
                            
                            # Add ride cymbal
                            e += [ec().intersect({"pitch": {"51 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add percussion instruments
                            percussion_pitches = {"56 (Drums)", "60 (Drums)", "61 (Drums)", "62 (Drums)", "63 (Drums)", "64 (Drums)"}
                            e += [ec().intersect({"pitch": percussion_pitches}).force_active() for _ in range(12)]
                            
                            # Add some optional drum notes for variety
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tempo (assuming we want to keep the original tempo)
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag (assuming we want to keep the original tag)
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e