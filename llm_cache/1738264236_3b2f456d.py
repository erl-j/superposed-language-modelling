# write a drum beat
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a basic drum beat with kick, snare, and hi-hat.
                            '''
                            e = []
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Kick drum on beats 1 and 3
                            for bar in range(4):
                                for beat in [0, 2]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"36 (Drums)"}, "instrument": {"Drums"}})
                                        .intersect({"onset/global_tick": {str(beat * 24 + bar * 96)}})
                                        .force_active()
                                    ]
                            
                            # Snare on beats 2 and 4
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "instrument": {"Drums"}})
                                        .intersect({"onset/global_tick": {str(beat * 24 + bar * 96)}})
                                        .force_active()
                                    ]
                            
                            # Hi-hat on every eighth note
                            for bar in range(4):
                                for eighth in range(8):
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}, "instrument": {"Drums"}})
                                        .intersect({"onset/global_tick": {str(eighth * 12 + bar * 96)}})
                                        .force_active()
                                    ]
                            
                            # Add up to 10 optional drum hits for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo (let's use a moderate tempo of 120 BPM)
                            e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]
                            
                            # Set tag (let's use 'pop' as a generic tag)
                            e = [ev.intersect({"tag": {"pop"}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e