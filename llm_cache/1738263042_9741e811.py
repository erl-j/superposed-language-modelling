# add some guitar arpeggio with 8th notes
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a guitar arpeggio with 8th notes to the existing beat.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside existing instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Guitar"})]
                            
                            # Start with an empty list for new events
                            e = []
                            
                            # Add guitar arpeggio with 8th notes
                            for i in range(32):  # 32 eighth notes in 4 bars
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Guitar"},
                                        "onset/global_tick": {str(i * 12)},  # 12 ticks per 8th note (24 ticks per quarter note)
                                        "duration": {"1/8"},  # 8th note duration
                                        "pitch": {str(pitch) for pitch in range(60, 85)}  # Mid-range guitar pitches
                                    })
                                    .force_active()
                                )
                            
                            # Add up to 10 optional guitar notes for variation
                            e += [ec().intersect({"instrument": {"Guitar"}}) for _ in range(10)]
                            
                            # Set tempo to current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e