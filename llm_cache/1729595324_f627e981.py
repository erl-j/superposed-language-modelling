# Add a bassline
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a bassline to the existing beat. The bassline will follow the rhythm of the kick drum
                            and add some additional notes for variation.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Find kick drum hits
                            kicks = [ev for ev in e if {"35 (Drums)", "36 (Drums)"}.intersection(ev.a["pitch"])]
                            
                            # Add bass notes that align with kick drum hits
                            for kick in kicks:
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": kick.a["onset/beat"],
                                        "onset/tick": kick.a["onset/tick"],
                                        "pitch": {str(pitch) for pitch in range(30, 55)},  # Bass range
                                    })
                                    .force_active()
                                ]
                            
                            # Add up to 8 additional bass notes for variation
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(pitch) for pitch in range(30, 55)},  # Bass range
                                })
                                for _ in range(8)
                            ]
                            
                            # Set velocity for bass notes (medium-high)
                            e = [ev.intersect(ec().velocity_constraint(80)) if ev.a["instrument"] == {"Bass"} else ev for ev in e]
                            
                            # Preserve the original tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the original tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e