# add a bass line that lines up with the kicks
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a bass line that aligns with the kick drum pattern.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Find kicks
                            kicks = [
                                ev
                                for ev in e_other
                                if {"35 (Drums)", "36 (Drums)", "37 (Drums)"}.intersection(ev.a["pitch"])
                            ]
                            
                            # Start building the new constraint
                            e = []
                            
                            # Add bass notes aligned with kicks
                            for kick in kicks:
                                e += [
                                    ec()
                                    .intersect(
                                        {
                                            "instrument": {"Bass"},
                                            "onset/global_tick": kick.a["onset/global_tick"],
                                            "pitch": {str(pitch) for pitch in range(30, 55)},  # Bass pitch range
                                            "duration": {"1/8", "1/4", "1/2"}  # Typical bass note durations
                                        }
                                    )
                                    .force_active()
                                ]
                            
                            # Add up to 8 additional bass notes for variation
                            e += [
                                ec()
                                .intersect(
                                    {
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in range(30, 55)}
                                    }
                                ) 
                                for _ in range(8)
                            ]
                            
                            # Set velocity for bass notes (medium-high for prominence)
                            e = [ev.intersect(ec().velocity_constraint(80)) for ev in e]
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e