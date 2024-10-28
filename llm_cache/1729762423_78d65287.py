# Create a rhythm using hemiolas
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a rhythm using hemiolas, which are rhythmic patterns that create a sense of 
                            two against three or three against two. We'll use drums to create this effect.
                            '''
                            e = []
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Create a 3/4 pattern over 4/4 time to create the hemiola effect
                            for bar in range(2):  # We'll create the pattern over two bars
                                for beat in range(3):  # 3 beats per hemiola cycle
                                    onset_beat = (bar * 4) + (beat * 2)  # This creates the 3 against 2 feel
                                    if onset_beat < 8:  # Ensure we don't go beyond 2 bars
                                        # Add kick drum
                                        e.append(
                                            ec()
                                            .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(onset_beat)}})
                                            .force_active()
                                        )
                                        # Add snare on the off-beat
                                        e.append(
                                            ec()
                                            .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/beat": {str(onset_beat)}, "onset/tick": {"12"}})
                                            .force_active()
                                        )
                            
                            # Add hi-hats to maintain the 4/4 feel
                            for beat in range(8):
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                )
                            
                            # Add some variation with open hi-hats
                            for beat in [2, 6]:
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"46 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                )
                            
                            # Add up to 10 optional drum hits for additional complexity
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo (preserving from input)
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag (preserving from input, adding 'hemiola')
                            e = [ev.intersect({"tag": {tag, "hemiola"}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e