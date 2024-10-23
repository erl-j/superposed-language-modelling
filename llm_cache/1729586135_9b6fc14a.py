# write a cumbia beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a cumbia beat with a characteristic rhythm pattern.
                            Cumbia typically has a tempo around 90-110 BPM, so we'll set it to 100.
                            We'll use kick, snare, hi-hat, and conga drums to create the rhythm.
                            '''
                            e = []
                            # Set tempo to 100 BPM
                            tempo_constraint = ec().tempo_constraint(100)
                            
                            # Kick drum pattern (on 1 and 3)
                            for bar in range(4):
                                for beat in [0, 2]:
                                    e.append(ec().intersect({
                                        "instrument": {"Drums"},
                                        "pitch": {"36 (Drums)"}, # Bass drum
                                        "onset/beat": {str(beat + bar * 4)},
                                        "onset/tick": {"0"}
                                    }).intersect(tempo_constraint).force_active())
                            
                            # Snare drum pattern (on 2 and 4)
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e.append(ec().intersect({
                                        "instrument": {"Drums"},
                                        "pitch": {"38 (Drums)"}, # Snare drum
                                        "onset/beat": {str(beat + bar * 4)},
                                        "onset/tick": {"0"}
                                    }).intersect(tempo_constraint).force_active())
                            
                            # Hi-hat pattern (eighth notes)
                            for bar in range(4):
                                for beat in range(4):
                                    for tick in [0, 12]:
                                        e.append(ec().intersect({
                                            "instrument": {"Drums"},
                                            "pitch": {"42 (Drums)"}, # Closed Hi-hat
                                            "onset/beat": {str(beat + bar * 4)},
                                            "onset/tick": {str(tick)}
                                        }).intersect(tempo_constraint).force_active())
                            
                            # Conga pattern (characteristic of cumbia)
                            for bar in range(4):
                                e.append(ec().intersect({
                                    "instrument": {"Drums"},
                                    "pitch": {"63 (Drums)"}, # High Conga
                                    "onset/beat": {str(bar * 4)},
                                    "onset/tick": {"16"}
                                }).intersect(tempo_constraint).force_active())
                                
                                e.append(ec().intersect({
                                    "instrument": {"Drums"},
                                    "pitch": {"64 (Drums)"}, # Low Conga
                                    "onset/beat": {str(bar * 4 + 1)},
                                    "onset/tick": {"16"}
                                }).intersect(tempo_constraint).force_active())
                            
                            # Add some variation
                            for _ in range(10):
                                e.append(ec().intersect({
                                    "instrument": {"Drums"},
                                    "pitch": {"42 (Drums)", "46 (Drums)", "63 (Drums)", "64 (Drums)"} # Various percussion
                                }).intersect(tempo_constraint))
                            
                            # Set tag to latino (closest to cumbia in the available tags)
                            tag_constraint = {"tag": {"latino"}}
                            e = [ev.intersect(tag_constraint) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e