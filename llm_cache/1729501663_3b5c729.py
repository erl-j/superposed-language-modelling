# Replace the bass with something more simple
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a simpler bassline, we'll create a pattern that emphasizes the 1 and 3 of each bar,
                            with occasional notes on the 2 and 4 for variation.
                            '''
                            # Remove inactive notes and existing bass notes
                            e = [ev for ev in e if ev.is_active() and ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Create a simple bassline
                            for bar in range(4):
                                # Bass on the 1 of each bar
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Bass"}, "onset/beat": {str(bar * 4)}})
                                    .force_active()
                                )
                                
                                # Bass on the 3 of each bar
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Bass"}, "onset/beat": {str(bar * 4 + 2)}})
                                    .force_active()
                                )
                                
                                # 50% chance of bass on 2 and 4 for variation
                                if bar % 2 == 0:
                                    e.append(
                                        ec()
                                        .intersect({"instrument": {"Bass"}, "onset/beat": {str(bar * 4 + 1)}})
                                        .force_active()
                                    )
                                else:
                                    e.append(
                                        ec()
                                        .intersect({"instrument": {"Bass"}, "onset/beat": {str(bar * 4 + 3)}})
                                        .force_active()
                                    )
            
                            # Add up to 4 optional bass notes for additional variation
                            e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(4)]
                            
                            # Ensure all bass notes are in a funky range (E1 to E3)
                            e = [ev.intersect({"pitch": set(map(str, range(28, 53)))}) if "Bass" in ev.a["instrument"] else ev for ev in e]
                            
                            # Set tempo to 96 (typical for funk)
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            
                            # Set tag to funk
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e