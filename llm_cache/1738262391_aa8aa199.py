# make a syncopated bass line
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a syncopated bass line that adds groove to the existing beat.
                            Syncopation typically involves emphasizing off-beats or weak beats.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Start over with bass line
                            e = []
                            
                            # Create syncopated bass pattern
                            # We'll place bass notes on the "and" of beats 1 and 3 in each bar
                            for bar in range(4):
                                for beat in [0, 2]:  # Beats 1 and 3
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/global_tick": {str(bar * 96 + beat * 24 + 12)},  # +12 ticks for the "and"
                                            "pitch": {str(pitch) for pitch in range(30, 55)},  # Bass range
                                            "duration": {"1/8", "1/4"}  # Short to medium notes for groove
                                        })
                                        .force_active()
                                    ]
                            
                            # Add some optional bass notes for variation
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(pitch) for pitch in range(30, 55)},
                                    "duration": {"1/16", "1/8", "1/4"}
                                }) 
                                for _ in range(8)
                            ]
                            
                            # Preserve tempo from existing beat
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve tag from existing beat
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e