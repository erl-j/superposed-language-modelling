# add some bass line with 8th notes
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a bassline with 8th notes to the existing beat.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Start with an empty list for new events
                            e = []
                            
                            # Add bass notes on every 8th note (12 ticks)
                            for tick in range(0, 384, 12):  # 384 ticks in 4 bars, 12 ticks per 8th note
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/global_tick": {str(tick)},
                                        "pitch": {str(pitch) for pitch in range(30, 55)},  # Bass range
                                        "duration": {"1/8"}  # 8th note duration
                                    })
                                    .force_active()
                                ]
                            
                            # Add up to 5 optional bass notes for variation
                            e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(5)]
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e