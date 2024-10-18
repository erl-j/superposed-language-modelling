# drill rap beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a drill rap beat, we'll focus on a dark, aggressive sound with heavy 808 bass, punchy kicks, and rapid hi-hats.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add 808 bass (low pitch, long duration)
                            e += [ec().intersect({"pitch": {"35 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add kicks (usually on the 1 and in between)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}})
                                    .intersect({"onset/beat": {str(bar * 4)}})
                                    .force_active()
                                ]
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}})
                                    .intersect({"onset/beat": {str(bar * 4 + 2.5)}})
                                    .force_active()
                                ]
            
                            # Add snares/claps (typically on 3)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)", "39 (Drums)"}})
                                    .intersect({"onset/beat": {str(bar * 4 + 2)}})
                                    .force_active()
                                ]
            
                            # Add rapid hi-hats (16th notes)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}})
                                    .intersect({"onset/beat": {str(beat * 0.25)}})
                                    .force_active()
                                ]
            
                            # Add some open hi-hats for variation
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
            
                            # Add some tom fills
                            e += [ec().intersect({"pitch": {"45 (Drums)", "47 (Drums)", "48 (Drums)"}}).force_active() for _ in range(6)]
            
                            # Set a typical drill tempo (around 140 BPM)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
            
                            # Set tag to electronic and hip-hop
                            e = [ev.intersect({"tag": {"electronic", "hip-hop", "-"}}) for ev in e]
            
                            # Add optional drum notes for more variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e