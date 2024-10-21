# Write a reggae beat.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a reggae beat, we'll focus on the characteristic one-drop rhythm,
                            with emphasis on the 3rd beat of each bar, and incorporate other typical reggae elements.
                            '''
                            e = []
                            # Set tempo to a typical reggae tempo (60-90 BPM)
                            tempo = 72
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Kick drum (emphasizing the 3rd beat of each bar)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(2 + bar * 4)}})
                                    .force_active()
                                ]
                            
                            # Snare (rim shot on beats 2 and 4)
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"40 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Hi-hat (eighth notes throughout)
                            for bar in range(4):
                                for beat in range(4):
                                    for tick in [0, 12]:
                                        e += [
                                            ec()
                                            .intersect({
                                                "pitch": {"42 (Drums)"},
                                                "onset/beat": {str(beat + bar * 4)},
                                                "onset/tick": {str(tick)}
                                            })
                                            .force_active()
                                        ]
                            
                            # Percussion (conga or bongo hits)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"60 (Drums)", "61 (Drums)"}, "onset/beat": {str(bar * 4)}})
                                    .force_active()
                                ]
                            
                            # Skank guitar or keyboard (offbeat chords)
                            for bar in range(4):
                                for beat in range(4):
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Electric Guitar", "Electric Piano"},
                                            "onset/beat": {str(beat + bar * 4)},
                                            "onset/tick": {"12"}
                                        })
                                        .force_active()
                                    ]
                            
                            # Bass (playing on the 1 and dropping on the 3)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Bass"}, "onset/beat": {str(bar * 4)}})
                                    .force_active()
                                ]
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Bass"}, "onset/beat": {str(1 + bar * 4)}})
                                    .force_active()
                                ]
                            
                            # Set tag to reggae
                            e = [ev.intersect({"tag": {"reggae-ska", "-"}}) for ev in e]
                            
                            # Add some optional percussion or melodic elements
                            e += [ec().intersect({"instrument": {"Drums", "Percussion"}}) for _ in range(10)]
                            e += [ec().intersect({"instrument": {"Electric Guitar", "Electric Piano", "Bass"}}) for _ in range(10)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e