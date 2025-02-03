# add some guitar chords
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add guitar chords to the existing beat. We'll add some basic chord patterns
                            that complement the rhythm without overpowering it.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Guitar"})]
                            
                            # Start with existing events
                            e = e_other.copy()
                            
                            # Add guitar chords
                            # We'll add 8 chord hits, two per bar
                            for bar in range(4):
                                for beat in [0, 2]:  # Add chords on the 1 and 3 of each bar
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Guitar"},
                                            "onset/global_tick": {str(beat * 24 + bar * 96)},  # 24 ticks per beat, 96 ticks per bar
                                            "duration": {"1/4", "1/2"},  # Quarter or half note duration
                                            "pitch": {str(pitch) for pitch in range(48, 72)}  # Mid-range guitar pitches
                                        })
                                        .force_active()
                                    ]
                            
                            # Add up to 4 optional guitar notes for variation
                            e += [ec().intersect({"instrument": {"Guitar"}}) for _ in range(4)]
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e