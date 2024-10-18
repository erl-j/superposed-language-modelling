# simple swing beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a simple swing beat, we'll focus on a basic pattern with kick, snare, and ride cymbal.
                            The swing feel will be created by slightly delaying the off-beat hits.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set up a two-bar pattern (8 beats)
                            for bar in range(2):
                                for beat in range(4):
                                    # Kick drum on beats 1 and 3
                                    if beat % 2 == 0:
                                        e.append(
                                            ec()
                                            .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                            .force_active()
                                        )
                                    
                                    # Snare drum on beats 2 and 4
                                    if beat % 2 == 1:
                                        e.append(
                                            ec()
                                            .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                            .force_active()
                                        )
                                    
                                    # Ride cymbal on every beat
                                    e.append(
                                        ec()
                                        .intersect({"pitch": {"51 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    )
                                    
                                    # Ride cymbal on the "and" of each beat, slightly delayed for swing feel
                                    e.append(
                                        ec()
                                        .intersect({"pitch": {"51 (Drums)"}, "onset/beat": {str(beat + bar * 4)}, "onset/tick": {"14"}})
                                        .force_active()
                                    )
            
                            # Set tempo to 120 BPM (a common tempo for swing)
                            e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]
                            
                            # Set tag to swing
                            e = [ev.intersect({"tag": {"swing", "-"}}) for ev in e]
                            
                            # Add some optional ghost notes
                            e += [ec().intersect({"instrument": {"Drums"}, "velocity": {"0-40"}}) for _ in range(10)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e