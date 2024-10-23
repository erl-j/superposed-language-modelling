# replace the drums with something simpler
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a simpler drum pattern, focusing on basic elements.
                            '''
                            # Remove all existing drum events
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Create a simple drum pattern
                            for bar in range(4):
                                for beat in range(4):
                                    # Kick drum on every beat
                                    e.append(
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(bar*4 + beat)}})
                                        .force_active()
                                    )
                                    
                                    # Snare on beats 2 and 4
                                    if beat % 2 != 0:
                                        e.append(
                                            ec()
                                            .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/beat": {str(bar*4 + beat)}})
                                            .force_active()
                                        )
                            
                            # Add hi-hats on every eighth note
                            for eighth in range(32):
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/beat": {str(eighth // 2)}, "onset/tick": {str(12 * (eighth % 2))}})
                                    .force_active()
                                )
                            
                            # Preserve the tempo from the original events
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the tag from the original events
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to maintain the total number of events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e