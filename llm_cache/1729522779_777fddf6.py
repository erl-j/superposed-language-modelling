# add a bass line with current tag
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a bass line that complements the existing funk beat, 
                            while maintaining the current tag and tempo.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Add a funky bass line
                            for bar in range(4):
                                for beat in range(4):
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/beat": {str(beat + bar * 4)},
                                            "pitch": {str(pitch) for pitch in range(35, 55)},  # Bass range
                                            "offset/beat": {str(beat + bar * 4 + 1)},  # Duration of 1 beat
                                        })
                                        .force_active()
                                    ]
                            
                            # Add some syncopation
                            for _ in range(4):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/tick": {"12"},  # Offbeat
                                        "pitch": {str(pitch) for pitch in range(35, 55)},
                                    })
                                    .force_active()
                                ]
                            
                            # Add up to 8 optional bass notes for variation
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Bass"},
                                    "pitch": {str(pitch) for pitch in range(35, 55)},
                                })
                                for _ in range(8)
                            ]
                            
                            # Set velocity for bass notes
                            e = [ev.intersect(ec().velocity_constraint(80)) if ev.a["instrument"] == {"Bass"} else ev for ev in e]
                            
                            # Maintain the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to current tag
                            e = [ev.intersect({"tag": tag}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e