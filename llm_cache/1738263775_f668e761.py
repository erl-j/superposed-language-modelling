# add a tom fill at the end of the drum beat
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a tom fill at the end of the drum beat.
                            We'll use high, mid, and low toms for this fill.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside non-drum instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Find existing drum events
                            e_drums = [ev for ev in e if not ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Define tom pitches
                            high_tom = "50 (Drums)"
                            mid_tom = "47 (Drums)"
                            low_tom = "45 (Drums)"
                            
                            # Create tom fill in the last quarter of the loop
                            fill_start = 3 * 96  # Start at the beginning of the last bar (assuming 4/4 time)
                            
                            # Add high tom
                            e.append(ec().intersect({
                                "instrument": {"Drums"},
                                "pitch": {high_tom},
                                "onset/global_tick": {str(fill_start)}
                            }).force_active())
                            
                            # Add mid tom
                            e.append(ec().intersect({
                                "instrument": {"Drums"},
                                "pitch": {mid_tom},
                                "onset/global_tick": {str(fill_start + 24)}  # 1/4 beat later
                            }).force_active())
                            
                            # Add low tom
                            e.append(ec().intersect({
                                "instrument": {"Drums"},
                                "pitch": {low_tom},
                                "onset/global_tick": {str(fill_start + 48)}  # 1/2 beat later
                            }).force_active())
                            
                            # Add a final hit on the high tom
                            e.append(ec().intersect({
                                "instrument": {"Drums"},
                                "pitch": {high_tom},
                                "onset/global_tick": {str(fill_start + 72)}  # 3/4 beat later
                            }).force_active())
                            
                            # Add back other instruments and existing drum events
                            e += e_other + e_drums
                            
                            # Set tempo to current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e