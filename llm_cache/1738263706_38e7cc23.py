# add some guitar stabs
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add guitar stabs to the existing beat. Guitar stabs are typically short, 
                            rhythmic chords that punctuate the groove. We'll add them on off-beats to complement 
                            the existing rhythm.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside existing instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Guitar"})]
                            
                            # Start with existing events
                            e = e_other.copy()
                            
                            # Add guitar stabs
                            for bar in range(4):
                                for beat in [0.5, 1.5, 2.5, 3.5]:  # Off-beats
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Guitar"},
                                            "onset/global_tick": {str(int(beat * 24 + bar * 4 * 24))},
                                            "duration": {"1/16", "1/8"},  # Short duration for stabs
                                            "pitch": {str(pitch) for pitch in range(50, 70)}  # Mid-range pitches
                                        })
                                        .intersect(ec().velocity_constraint(80))  # Moderately loud
                                        .force_active()
                                    ]
                            
                            # Add a few optional guitar notes for variation
                            e += [ec().intersect({"instrument": {"Guitar"}}) for _ in range(5)]
                            
                            # Preserve tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e