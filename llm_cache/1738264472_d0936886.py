# regenerate the pitches in the second half
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to regenerate the pitches in the second half of the beat while keeping the rhythm and other attributes intact.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Split the events into first and second half
                            first_half = [ev for ev in e if int(list(ev.a["onset/global_tick"])[0]) < 192]  # 192 ticks is the midpoint of 4 bars
                            second_half = [ev for ev in e if int(list(ev.a["onset/global_tick"])[0]) >= 192]
                            
                            # For the second half, we'll remove the pitch constraint and let the model regenerate it
                            new_second_half = []
                            for ev in second_half:
                                new_ev = ec()
                                for attr, value in ev.a.items():
                                    if attr != "pitch":
                                        new_ev = new_ev.intersect({attr: value})
                                new_second_half.append(new_ev)
                            
                            # Combine first half (unchanged) and new second half
                            e = first_half + new_second_half
                            
                            # Ensure tempo and tag are preserved
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to maintain the total number of events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e