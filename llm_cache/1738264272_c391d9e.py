# Add a syncopated bass line
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a syncopated bass line to the existing beat.
                            Syncopation typically involves emphasizing off-beats or weak beats.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Start with an empty list for new events
                            e = []
                            
                            # Add syncopated bass notes
                            # We'll place bass notes on the "and" of beats 1, 2, and 3, and on beat 4
                            syncopated_ticks = [12, 36, 60, 72]  # 12 ticks = 1/2 beat
                            for bar in range(4):
                                for tick in syncopated_ticks:
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/global_tick": {str(tick + bar * 96)},  # 96 ticks per bar
                                            "pitch": {str(pitch) for pitch in range(30, 55)},  # Bass range
                                            "duration": {"1/8", "1/4"}  # Short to medium notes for punchiness
                                        })
                                        .force_active()
                                    ]
                            
                            # Add a few more optional bass notes for variety
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(pitch) for pitch in range(30, 55)}
                                }) 
                                for _ in range(5)
                            ]
                            
                            # Set velocity to be moderately high for emphasis
                            e = [ev.intersect(ec().velocity_constraint(100)) for ev in e]
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e