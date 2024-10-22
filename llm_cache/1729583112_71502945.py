# Add a groovy bassline
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a groovy bassline that complements the existing rhythm.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Add a groovy bassline
                            for bar in range(4):
                                # Root note on the 1
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "onset/beat": {str(bar * 4)},
                                    "pitch": {str(pitch) for pitch in range(36, 48)},  # E1 to B1
                                }).force_active())
                                
                                # Syncopated notes
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "onset/beat": {str(bar * 4 + 1)},
                                    "onset/tick": {"12"},  # On the "and" of 2
                                    "pitch": {str(pitch) for pitch in range(36, 48)},
                                }).force_active())
                                
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "onset/beat": {str(bar * 4 + 2)},
                                    "onset/tick": {"18"},  # On the "a" of 3
                                    "pitch": {str(pitch) for pitch in range(36, 48)},
                                }).force_active())
                                
                                # Fill on the 4
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "onset/beat": {str(bar * 4 + 3)},
                                    "pitch": {str(pitch) for pitch in range(36, 48)},
                                }).force_active())
                                
                            # Add some optional notes for variation
                            for _ in range(8):
                                e.append(ec().intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(pitch) for pitch in range(36, 48)},
                                }))
                            
                            # Set velocity for dynamic playing
                            e = [ev.intersect(ec().velocity_constraint(80)) for ev in e if ev.a["instrument"] == {"Bass"}]
                            
                            # Preserve tempo and tag from input
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e