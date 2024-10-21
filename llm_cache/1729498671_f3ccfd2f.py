# A drum and bass beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a drum and bass beat, we'll focus on a fast-paced rhythm with prominent kick and snare patterns, 
                            along with rapid hi-hat or cymbal work. We'll aim for a tempo around 170 BPM, which is typical for drum and bass.
                            '''
                            e = []
                            # Clear existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set up a basic kick pattern (typically on the 1 and 3 or with syncopation)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}}).force_active(),
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}}).force_active()
                                ]
                            
                            # Add snares on the 2 and 4 of each bar
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + 1)}}).force_active(),
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + 3)}}).force_active()
                                ]
                            
                            # Add rapid hi-hats (closed)
                            for beat in range(16):
                                e += [ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}}).force_active()]
                                # Add some offbeat hi-hats
                                if beat % 2 == 0:
                                    e += [ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"12"}}).force_active()]
            
                            # Add some open hi-hats for variation
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
            
                            # Add some tom fills in the last bar
                            e += [
                                ec().intersect({"pitch": {"45 (Drums)", "47 (Drums)", "48 (Drums)"}, "onset/beat": {"12", "13", "14", "15"}}).force_active()
                                for _ in range(4)
                            ]
            
                            # Add up to 20 optional drum notes for additional complexity
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
            
                            # Set tempo to 170 BPM
                            e = [ev.intersect(ec().tempo_constraint(170)) for ev in e]
            
                            # Set tag to dnb (drum and bass)
                            e = [ev.intersect({"tag": {"dnb", "-"}}) for ev in e]
            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e