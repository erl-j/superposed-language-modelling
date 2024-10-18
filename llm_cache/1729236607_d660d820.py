# drill rap beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a drill rap beat, we'll focus on a dark, aggressive sound with heavy 808 bass, rapid hi-hats, and sparse snares.
                            '''
                            e = []
                            # remove all drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add 808 bass drum (usually pitched lower for drill)
                            e += [ec().intersect({"pitch": {"35 (Drums)"}}).force_active() for _ in range(16)]
                            
                            # Add snares (typically on beats 3 and 7 in a two-bar pattern)
                            for bar in range(2):
                                for beat in [2, 6]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 8)}})
                                        .force_active()
                                    ]
                            
                            # Add rapid hi-hats (characteristic of drill beats)
                            e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(32)]
                            
                            # Add some open hi-hats for variation
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add some claps to layer with snares
                            e += [ec().intersect({"pitch": {"39 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add up to 20 optional drum notes for additional complexity
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tempo to 140 BPM (typical for drill rap)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
                            
                            # Set tag to electronic and hip-hop
                            e = [ev.intersect({"tag": {"electronic", "hip-hop", "-"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events