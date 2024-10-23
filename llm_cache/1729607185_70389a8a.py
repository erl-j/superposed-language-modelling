# write a fast drum n bass drum beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a fast drum and bass beat with a typical breakbeat pattern,
                            featuring kicks, snares, and hi-hats. We'll aim for a high tempo and complex rhythms.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set a fast tempo typical for drum and bass (around 170-180 BPM)
                            tempo = 175
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Kick drum pattern (typically on the 1 and sometimes syncopated)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}}).force_active(),
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}, "onset/tick": {"12"}}).force_active()
                                ]
                            
                            # Snare drum (typically on the 2 and 4)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + 1)}}).force_active(),
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + 3)}}).force_active()
                                ]
                            
                            # Hi-hats (16th note pattern)
                            for beat in range(16):
                                e += [ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"0", "12"}}).force_active()]
                            
                            # Add some ghost notes and variations
                            e += [ec().intersect({"pitch": {"38 (Drums)", "40 (Drums)"}, "velocity": ec().velocity_constraint(40)}).force_active() for _ in range(8)]
                            
                            # Add some cymbal crashes or rides for accent
                            e += [ec().intersect({"pitch": {"49 (Drums)", "51 (Drums)"}, "onset/beat": {"0", "8"}}).force_active() for _ in range(2)]
                            
                            # Allow for some additional drum hits for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tag to drum and bass
                            e = [ev.intersect({"tag": {"dance-eletric"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e