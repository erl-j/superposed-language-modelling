# A drum and bass beat some snare rolls
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a drum and bass beat with snare rolls, we'll use kicks, snares, hi-hats, and some additional percussion.
                            We'll create a typical dnb pattern and add snare rolls in strategic places.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set up a basic dnb pattern
                            # Kicks (usually on the 1 and sometimes syncopated)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}})
                                    .force_active()
                                ]
                                # Add some syncopated kicks
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + 1.75)}})
                                    .force_active()
                                ]
                            
                            # Snares (typically on the 2 and 4)
                            for bar in range(4):
                                for beat in [2, 4]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + beat - 1)}})
                                        .force_active()
                                    ]
                            
                            # Hi-hats (16th note pattern)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"0", "6", "12", "18"}})
                                    .force_active()
                                ]
                            
                            # Add snare rolls
                            # Roll 1: 16th notes in the second half of bar 2
                            for tick in range(0, 24, 6):  # 16th notes
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {"6", "7"}, "onset/tick": {str(tick)}})
                                    .force_active()
                                ]
                            
                            # Roll 2: 32nd notes in the last beat of bar 4
                            for tick in range(0, 24, 3):  # 32nd notes
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {"15"}, "onset/tick": {str(tick)}})
                                    .force_active()
                                ]
                            
                            # Add some additional percussion (e.g., toms or cymbals)
                            e += [ec().intersect({"pitch": {"45 (Drums)", "49 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add up to 20 optional drum notes for variety
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tempo to 170 BPM (typical for drum and bass)
                            e = [ev.intersect(ec().tempo_constraint(170)) for ev in e]
                            
                            # Set tag to dnb (drum and bass)
                            e = [ev.intersect({"tag": {"dnb", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events