# Add kicks on the upbeat
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add kicks on the upbeat to the existing beat.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Start with existing drum events
                            e_drums = [ev for ev in e if not ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add kicks on the upbeat (2nd and 4th sixteenth notes of each beat)
                            for bar in range(4):
                                for beat in range(4):
                                    for sixteenth in [1, 3]:
                                        e_drums.append(
                                            ec()
                                            .intersect({
                                                "instrument": {"Drums"},
                                                "pitch": {"36 (Drums)"}, # Standard kick drum pitch
                                                "onset/global_tick": {str(bar * 96 + beat * 24 + sixteenth * 6)},
                                                "offset/global_tick": {"none (Drums)"},
                                                "duration": {"duration:none (Drums)"}
                                            })
                                            .force_active()
                                        )
                            
                            # Combine all events
                            e = e_drums + e_other
                            
                            # Set tempo to current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e