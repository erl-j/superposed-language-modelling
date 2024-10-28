# Remove the old piano and create some new piano chords that line up with the drum kicks
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to remove the old piano parts and create new piano chords that align with the drum kicks.
                            '''
                            # Remove inactive notes and old piano parts
                            e = [ev for ev in e if ev.is_active() and ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Find kicks
                            kicks = [
                                ev
                                for ev in e
                                if {"35 (Drums)", "36 (Drums)", "37 (Drums)"}.intersection(ev.a["pitch"])
                            ]
                            
                            # Create new piano chords aligned with kicks
                            for kick in kicks:
                                e += [
                                    ec()
                                    .intersect(
                                        {
                                            "instrument": {"Piano"},
                                            "onset/beat": kick.a["onset/beat"],
                                            "onset/tick": kick.a["onset/tick"],
                                            "pitch": {str(pitch) for pitch in range(60, 85)},  # Mid-range piano notes
                                        }
                                    )
                                    .force_active()
                                    for _ in range(3)  # Create 3 notes for each chord
                                ]
                            
                            # Add up to 10 optional piano notes for variation
                            e += [ec().intersect({"instrument": {"Piano"}}) for _ in range(10)]
                            
                            # Preserve tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e