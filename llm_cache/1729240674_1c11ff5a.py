# beat with rattling hats and 808s (35 midi pitch). Have some percussion as well in the last bar
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a beat with rattling hats, 808s, and percussion in the last bar.
                            '''
                            e = []
                            # remove all drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add 808 kicks (35 midi pitch)
                            e += [ec().intersect({"pitch": {"35 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add rattling hi-hats (using closed and open hats)
                            for i in range(32):  # 16th note grid for 2 bars
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)", "46 (Drums)"}})  # Closed and open hats
                                    .intersect({"onset/beat": {str(i // 4)}, "onset/tick": {str((i % 4) * 6)}})
                                    .force_active()
                                ]
                            
                            # Add some variation to hi-hats
                            e += [ec().intersect({"pitch": {"42 (Drums)", "46 (Drums)"}}).force_active() for _ in range(10)]
                            
                            # Add snares on 2 and 4
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}})
                                        .intersect({"onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add percussion in the last bar
                            last_bar_percussion = [
                                "39 (Drums)",  # Clap
                                "47 (Drums)",  # Low-Mid Tom
                                "50 (Drums)",  # High Tom
                                "56 (Drums)",  # Cowbell
                                "60 (Drums)",  # Hi Bongo
                                "61 (Drums)",  # Low Bongo
                                "62 (Drums)",  # Mute Hi Conga
                                "63 (Drums)",  # Open Hi Conga
                            ]
                            
                            for _ in range(12):  # Add 12 percussion hits in the last bar
                                e += [
                                    ec()
                                    .intersect({"pitch": set(last_bar_percussion)})
                                    .intersect({"onset/beat": {"12", "13", "14", "15"}})
                                    .force_active()
                                ]
                            
                            # Add some optional drum notes
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tempo to 90 BPM
                            e = [ev.intersect(ec().tempo_constraint(90)) for ev in e]
                            
                            # Set tag to electronic and groove
                            e = [ev.intersect({"tag": {"electronic", "groove"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events