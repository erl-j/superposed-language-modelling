# add a piano arpeggio
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a piano arpeggio to the existing beat.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside existing instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Start with an empty list for new events
                            e = []
                            
                            # Add piano arpeggio notes
                            for bar in range(4):
                                for beat in range(4):
                                    for sixteenth in range(4):
                                        e += [
                                            ec()
                                            .intersect({
                                                "instrument": {"Piano"},
                                                "onset/global_tick": {str(bar * 96 + beat * 24 + sixteenth * 6)},
                                                "pitch": {str(pitch) for pitch in range(60, 85)},  # Middle to high range for arpeggio
                                                "duration": {"1/16", "1/8"}  # Short durations for arpeggio effect
                                            })
                                            .force_active()
                                        ]
            
                            # Add up to 10 optional piano notes for variation
                            e += [ec().intersect({"instrument": {"Piano"}}) for _ in range(10)]
                            
                            # Set tempo to current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e