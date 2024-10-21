# Write a drum and bass beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a drum and bass beat, we'll create a fast-paced rhythm with a prominent bassline,
                            breakbeat-style drums, and a tempo around 170-180 BPM.
                            '''
                            e = []
                            # Set tempo to 174 BPM (typical for drum and bass)
                            tempo_constraint = ec().tempo_constraint(174)
                            
                            # Drum pattern
                            # Kick drum (usually on the 1 and sometimes on the 3)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}}).force_active(),
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}})
                                ]
                            
                            # Snare (typically on the 2 and 4)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + 1)}}).force_active(),
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + 3)}}).force_active()
                                ]
                            
                            # Hi-hats (16th note pattern)
                            for beat in range(16):
                                e += [ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"0", "12"}}).force_active()]
                            
                            # Add some ghost notes and variations
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Bassline
                            # Create a rolling bassline with 16th note variations
                            for beat in range(16):
                                e += [
                                    ec().intersect({"instrument": {"Bass"}, "onset/beat": {str(beat)}}).force_active(),
                                    ec().intersect({"instrument": {"Bass"}, "onset/beat": {str(beat)}, "onset/tick": {"12"}})
                                ]
                            
                            # Add some optional bass notes for variation
                            e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(10)]
                            
                            # Apply tempo constraint to all events
                            e = [ev.intersect(tempo_constraint) for ev in e]
                            
                            # Set tag to dance-eletric (closest to drum and bass)
                            e = [ev.intersect({"tag": {"dance-eletric", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e