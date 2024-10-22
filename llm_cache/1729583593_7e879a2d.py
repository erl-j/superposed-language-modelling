# add some syncopated claps
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add syncopated claps to the existing beat. Syncopation typically involves placing
                            emphasis on the off-beats or weak beats. We'll add claps on the "and" counts (the 8th notes
                            between beats) and some on the last 16th note of certain beats.
                            '''
                            # Keep all existing events
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Add syncopated claps
                            # Clap on the "and" of beat 2 and 4 in each bar
                            for bar in range(4):
                                for beat in [1, 3]:  # corresponds to beats 2 and 4
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Drums"},
                                            "pitch": {"39 (Drums)"},  # 39 is typically used for claps
                                            "onset/beat": {str(beat + bar * 4)},
                                            "onset/tick": {"12"}  # 12 ticks is halfway between beats (remember, 24 ticks per beat)
                                        })
                                        .force_active()
                                    )
            
                            # Add some claps on the last 16th note of beats 1 and 3 in bars 2 and 4
                            for bar in [1, 3]:
                                for beat in [0, 2]:  # corresponds to beats 1 and 3
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Drums"},
                                            "pitch": {"39 (Drums)"},
                                            "onset/beat": {str(beat + bar * 4)},
                                            "onset/tick": {"18"}  # 18 ticks is the last 16th note of a beat
                                        })
                                        .force_active()
                                    )
            
                            # Add a few optional claps for variety
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Drums"},
                                    "pitch": {"39 (Drums)"}
                                })
                                for _ in range(5)
                            ]
            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
            
                            # Pad with inactive events to reach n_events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e