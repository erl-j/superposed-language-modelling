# Add some syncopated toms at the end of the 2nd and 4th bar
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add syncopated toms at the end of the 2nd and 4th bar.
                            Toms are typically represented by MIDI notes 45, 47, and 50 for low, mid, and high toms respectively.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Start with existing drum events
                            e_drums = [ev for ev in e if not ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add syncopated toms at the end of the 2nd bar (beats 6-7)
                            e_drums += [
                                ec().intersect({
                                    "instrument": {"Drums"},
                                    "pitch": {"45 (Drums)", "47 (Drums)", "50 (Drums)"},
                                    "onset/beat": {"6", "7"},
                                    "onset/tick": {"12", "18"} # Syncopated rhythm
                                }).force_active()
                                for _ in range(3) # Add 3 tom hits
                            ]
                            
                            # Add syncopated toms at the end of the 4th bar (beats 14-15)
                            e_drums += [
                                ec().intersect({
                                    "instrument": {"Drums"},
                                    "pitch": {"45 (Drums)", "47 (Drums)", "50 (Drums)"},
                                    "onset/beat": {"14", "15"},
                                    "onset/tick": {"12", "18"} # Syncopated rhythm
                                }).force_active()
                                for _ in range(3) # Add 3 tom hits
                            ]
                            
                            # Combine all events
                            e = e_drums + e_other
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to reach n_events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e