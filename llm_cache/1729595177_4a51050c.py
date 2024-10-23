# write an 909 dance beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a classic 909 dance beat with kick, snare, hi-hats, and some percussion.
                            The 909 is known for its punchy kick, crisp snare, and distinctive hi-hats.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Kick drum on every quarter note
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                    .intersect(ec().velocity_constraint(100))  # Strong kick
                                    .force_active()
                                ]
                            
                            # Snare on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .intersect(ec().velocity_constraint(90))
                                        .force_active()
                                    ]
                            
                            # Closed hi-hats on every 8th note
                            for beat in range(16):
                                for tick in [0, 12]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {str(tick)}})
                                        .intersect(ec().velocity_constraint(80))
                                        .force_active()
                                    ]
                            
                            # Open hi-hat on the off-beats
                            for beat in range(16):
                                if beat % 2 != 0:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"46 (Drums)"}, "onset/beat": {str(beat)}})
                                        .intersect(ec().velocity_constraint(85))
                                        .force_active()
                                    ]
                            
                            # Add some percussion (cowbell or clap) on the last beat of each bar
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"39 (Drums)", "56 (Drums)"}, "onset/beat": {str(3 + bar * 4)}})
                                    .intersect(ec().velocity_constraint(75))
                                    .force_active()
                                ]
                            
                            # Set tempo to 128 BPM (typical for dance music)
                            e = [ev.intersect(ec().tempo_constraint(128)) for ev in e]
                            
                            # Set tag to dance-eletric
                            e = [ev.intersect({"tag": {"dance-eletric"}}) for ev in e]
                            
                            # Add some optional drum hits for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e