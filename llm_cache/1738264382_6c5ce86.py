# add a piano melody
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a piano melody to the existing beat. The melody will be in a higher register
                            and will complement the existing rhythm without overpowering it.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside existing instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Start with existing events
                            e = e_other.copy()
                            
                            # Add piano melody
                            for _ in range(8):  # Add 8 piano notes
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "pitch": {str(pitch) for pitch in range(60, 85)},  # Higher register
                                        "duration": {"1/8", "1/4", "1/2"}  # Varying note lengths
                                    })
                                    .force_active()
                                ]
                            
                            # Add up to 4 more optional piano notes
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Piano"},
                                    "pitch": {str(pitch) for pitch in range(60, 85)}
                                }) 
                                for _ in range(4)
                            ]
                            
                            # Set velocity for piano notes (slightly quieter than the beat)
                            e = [ev.intersect(ec().velocity_constraint(70)) if "Piano" in ev.a["instrument"] else ev for ev in e]
                            
                            # Preserve tempo from existing events
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve tag from existing events
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e