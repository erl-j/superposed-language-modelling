# Add a dancy techno bassline
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a dancy techno bassline to the existing beat.
                            This will involve adding a repetitive, rhythmic bass pattern typical of techno music.
                            '''
                            # Remove inactive notes and any existing bass notes
                            e = [ev for ev in e if ev.is_active() and ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Create a techno bassline pattern
                            for bar in range(4):  # Assuming a 4-bar loop
                                for beat in range(4):  # 4 beats per bar
                                    # On-beat bass notes
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/beat": {str(bar * 4 + beat)},
                                            "onset/tick": {"0"},
                                            "pitch": {str(pitch) for pitch in range(36, 48)},  # Low to mid-low range
                                        })
                                        .intersect(ec().velocity_constraint(100))  # Strong velocity for emphasis
                                        .force_active()
                                    )
                                    
                                    # Off-beat bass notes for rhythm
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/beat": {str(bar * 4 + beat)},
                                            "onset/tick": {"12"},  # Halfway between beats
                                            "pitch": {str(pitch) for pitch in range(36, 48)},
                                        })
                                        .intersect(ec().velocity_constraint(80))  # Slightly lower velocity
                                        .force_active()
                                    )
                            
                            # Add some variation with optional notes
                            for _ in range(8):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in range(36, 52)},  # Slightly wider range for variation
                                    })
                                )
                            
                            # Preserve the original tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the original tag, but add 'dance-eletric' if not already present
                            original_tag = next(iter(e[0].a["tag"]))
                            new_tag = {"dance-eletric", original_tag} if original_tag != "dance-eletric" else {original_tag}
                            e = [ev.intersect({"tag": new_tag}) for ev in e]
                            
                            # Pad with inactive events to maintain the original number of events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e