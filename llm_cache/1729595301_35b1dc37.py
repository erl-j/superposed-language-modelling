# Add some syncopated toms to the beat

def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add syncopated toms to the existing beat. Syncopation means emphasizing the off-beats,
                            so we'll add toms on the "and" counts (the 8th notes between beats) and some 16th note offsets.
                            '''
                            # Keep all existing events
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Define tom drum pitches
                            tom_pitches = {"45 (Drums)", "47 (Drums)", "50 (Drums)"}  # Low, Mid, and High Tom
                            
                            # Add syncopated toms
                            for bar in range(4):  # Assuming a 4-bar loop
                                for beat in range(4):  # 4 beats per bar
                                    # Add tom on the "and" of each beat (8th note syncopation)
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Drums"},
                                            "pitch": tom_pitches,
                                            "onset/beat": {str(bar * 4 + beat)},
                                            "onset/tick": {"12"}  # 12 ticks is halfway between beats
                                        })
                                        .force_active()
                                    )
                                    
                                    # Add occasional 16th note syncopation
                                    if beat % 2 == 1:  # On odd beats for variety
                                        e.append(
                                            ec()
                                            .intersect({
                                                "instrument": {"Drums"},
                                                "pitch": tom_pitches,
                                                "onset/beat": {str(bar * 4 + beat)},
                                                "onset/tick": {"6", "18"}  # 6 and 18 ticks are 16th note offsets
                                            })
                                            .force_active()
                                        )
            
                            # Add some optional tom hits for more variety
                            e += [ec().intersect({"instrument": {"Drums"}, "pitch": tom_pitches}) for _ in range(8)]
                            
                            # Preserve the original tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the original tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to maintain the total number of events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e