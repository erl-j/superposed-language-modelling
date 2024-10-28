# Create some piano chords that line up with the drum kicks
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add piano chords that align with the drum kicks.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Start over with new events
                            e = []
                            
                            # Find kicks
                            kicks = [
                                ev
                                for ev in e_other
                                if {"35 (Drums)", "36 (Drums)", "37 (Drums)"}.intersection(ev.a["pitch"])
                            ]
                            
                            # Add piano chords on every kick
                            for kick in kicks:
                                e += [
                                    ec()
                                    .intersect(
                                        {
                                            "instrument": {"Piano"},
                                            "onset/beat": kick.a["onset/beat"],
                                            "onset/tick": kick.a["onset/tick"],
                                            "pitch": {str(pitch) for pitch in range(60, 85)}, # Set pitch range for piano chords
                                        }
                                    )
                                    .force_active()
                                    for _ in range(3)  # Add 3 notes for each chord
                                ]
                            
                            # Add up to 10 more optional piano notes for variation
                            e += [ec().intersect({"instrument": {"Piano"}}) for _ in range(10)]
                            
                            # Intersect with current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e