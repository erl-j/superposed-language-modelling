# add a rhythmic funky guitar 
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a rhythmic funky guitar to the existing beat.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove any existing guitar notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Guitar"})]
                            
                            # Add rhythmic guitar notes
                            for beat in range(16):  # 16 beats for a 4-bar loop
                                # Add a guitar chord on each beat
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Guitar"},
                                        "onset/beat": {str(beat)},
                                        "pitch": {str(pitch) for pitch in range(50, 65)},  # Mid-range pitches for guitar
                                    })
                                    .force_active()
                                ]
                                
                                # Add some syncopated notes
                                if beat % 2 == 0:  # On even beats
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Guitar"},
                                            "onset/beat": {str(beat)},
                                            "onset/tick": {"12"},  # Syncopated note on the "and" of the beat
                                            "pitch": {str(pitch) for pitch in range(50, 65)},
                                        })
                                        .force_active()
                                    ]
            
                            # Add some optional guitar notes for variation
                            e += [ec().intersect({"instrument": {"Guitar"}}) for _ in range(10)]
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set velocity for guitar to be moderately loud
                            e = [ev.intersect(ec().velocity_constraint(80)) if ev.a["instrument"] == {"Guitar"} else ev for ev in e]
                            
                            # Preserve the current tag (assuming it's "funk" based on the request)
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e