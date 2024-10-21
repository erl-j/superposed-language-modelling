# Create a metal beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a metal beat, we'll focus on creating a fast, aggressive drum pattern with double bass, 
                            frequent crash cymbals, and a mix of snare hits. We'll aim for a tempo around 180 BPM.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Double bass drum pattern (alternating 16th notes)
                            for beat in range(16):
                                for tick in [0, 12]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {str(tick)}})
                                        .force_active()
                                    ]
                            
                            # Snare on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Crash cymbal on the 1 of each bar
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"49 (Drums)"}, "onset/beat": {str(bar * 4)}})
                                    .force_active()
                                ]
                            
                            # Hi-hat pattern (8th notes)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"0"}})
                                    .force_active()
                                ]
                            
                            # Add some random tom fills
                            e += [ec().intersect({"pitch": {"45 (Drums)", "47 (Drums)", "48 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add some additional crash cymbals
                            e += [ec().intersect({"pitch": {"49 (Drums)", "57 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Set tempo to 180 BPM
                            e = [ev.intersect(ec().tempo_constraint(180)) for ev in e]
                            
                            # Set tag to metal
                            e = [ev.intersect({"tag": {"metal", "-"}}) for ev in e]
                            
                            # Add up to 20 optional drum notes for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e