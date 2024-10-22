# Rewrite the drums as a dance beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a dance beat with a four-on-the-floor kick pattern, 
                            snares on 2 and 4, and a consistent hi-hat pattern.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Four-on-the-floor kick pattern
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Snares on 2 and 4
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Hi-hat pattern (closed hi-hat on every 8th note)
                            for beat in range(16):
                                for tick in [0, 12]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {str(tick)}})
                                        .force_active()
                                    ]
                            
                            # Add some open hi-hats for variation
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"46 (Drums)"}, "onset/beat": {str(3 + bar * 4)}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
                            
                            # Add up to 10 optional percussion hits for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo to a typical dance tempo (128 BPM)
                            e = [ev.intersect(ec().tempo_constraint(128)) for ev in e]
                            
                            # Set tag to dance-eletric
                            e = [ev.intersect({"tag": {"dance-eletric"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e