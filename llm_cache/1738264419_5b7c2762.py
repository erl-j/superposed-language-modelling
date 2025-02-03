# change the pitches of the second half
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to change the pitches of the second half of the beat while keeping the rhythm and other attributes intact.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Separate the events into first and second half
                            first_half = [ev for ev in e if int(list(ev.a["onset/global_tick"])[0]) < 192]  # 192 ticks is the midpoint (2 bars)
                            second_half = [ev for ev in e if int(list(ev.a["onset/global_tick"])[0]) >= 192]
                            
                            # Keep the first half as is
                            new_e = first_half.copy()
                            
                            # Modify the second half
                            for ev in second_half:
                                new_ev = ec()
                                # Keep all attributes except pitch
                                for attr, value in ev.a.items():
                                    if attr != "pitch":
                                        new_ev = new_ev.intersect({attr: value})
                                
                                # For non-drum instruments, change the pitch
                                if ev.a["instrument"].isdisjoint({"Drums"}):
                                    # Shift the pitch by a random amount within a reasonable range
                                    new_ev = new_ev.intersect({"pitch": {str(pitch) for pitch in range(pitch_range[0], pitch_range[1]+1)}})
                                else:
                                    # For drums, we might want to change to a different drum sound
                                    new_ev = new_ev.intersect({"pitch": {drum for drum in drums}})
                                
                                new_ev = new_ev.force_active()
                                new_e.append(new_ev)
            
                            # Preserve tempo and tag
                            new_e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in new_e]
                            new_e = [ev.intersect({"tag": {tag}}) for ev in new_e]
            
                            # Pad with inactive events if necessary
                            new_e += [ec().force_inactive() for _ in range(n_events - len(new_e))]
            
                            return new_e