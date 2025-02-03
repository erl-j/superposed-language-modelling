# add a half time drum beat
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a half-time drum beat, which typically emphasizes beats 1 and 3 in a 4/4 time signature.
                            We'll keep other instruments intact and add our new drum pattern.
                            '''
                            # Remove inactive notes and set aside non-drum instruments
                            e = [ev for ev in e if ev.is_active()]
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Start with an empty list for our new events
                            e = []
                            
                            # Add kick drum on beats 1 and 3 of each bar
                            for bar in range(4):
                                for beat in [0, 2]:  # 0 and 2 for beats 1 and 3
                                    e.append(
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/global_tick": {str(bar * 96 + beat * 48)}})
                                        .force_active()
                                    )
                            
                            # Add snare drum on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:  # 1 and 3 for beats 2 and 4
                                    e.append(
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/global_tick": {str(bar * 96 + beat * 48)}})
                                        .force_active()
                                    )
                            
                            # Add hi-hat on every beat
                            for bar in range(4):
                                for beat in range(4):
                                    e.append(
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/global_tick": {str(bar * 96 + beat * 24)}})
                                        .force_active()
                                    )
                            
                            # Add some ghost notes and variations (optional drum hits)
                            for _ in range(10):
                                e.append(ec().intersect({"instrument": {"Drums"}}))
                            
                            # Preserve the original tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the original tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e