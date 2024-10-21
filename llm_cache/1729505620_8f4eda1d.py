# Add some drums
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a more complex drum pattern, we'll add various drum elements
                            while maintaining the existing funk beat structure.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Add kick drum pattern
                            for bar in range(4):
                                for beat in [0, 2, 3]:  # Emphasize 1 and 3, with an extra kick on 4
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add snare on 2 and 4
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add hi-hat pattern (closed hi-hat)
                            for bar in range(4):
                                for beat in range(4):
                                    for tick in [0, 12]:  # 8th note pattern
                                        e += [
                                            ec()
                                            .intersect({
                                                "pitch": {"42 (Drums)"},
                                                "onset/beat": {str(beat + bar * 4)},
                                                "onset/tick": {str(tick)}
                                            })
                                            .force_active()
                                        ]
                            
                            # Add open hi-hat occasionally
                            for bar in [1, 3]:  # Add open hi-hat in 2nd and 4th bars
                                e += [
                                    ec()
                                    .intersect({
                                        "pitch": {"46 (Drums)"},
                                        "onset/beat": {str(2 + bar * 4)},
                                        "onset/tick": {"12"}
                                    })
                                    .force_active()
                                ]
                            
                            # Add tom fills in the last bar
                            tom_pitches = ["45 (Drums)", "47 (Drums)", "50 (Drums)"]  # Low, mid, high toms
                            for beat in [12, 13, 14]:
                                for tick in [0, 8, 16]:
                                    e += [
                                        ec()
                                        .intersect({
                                            "pitch": set(tom_pitches),
                                            "onset/beat": {str(beat)},
                                            "onset/tick": {str(tick)}
                                        })
                                        .force_active()
                                    ]
                            
                            # Add crash cymbal at the beginning
                            e += [
                                ec()
                                .intersect({"pitch": {"49 (Drums)"}, "onset/beat": {"0"}, "onset/tick": {"0"}})
                                .force_active()
                            ]
                            
                            # Add ride cymbal pattern in the 3rd bar
                            for beat in range(8, 12):
                                e += [
                                    ec()
                                    .intersect({
                                        "pitch": {"51 (Drums)"},
                                        "onset/beat": {str(beat)},
                                        "onset/tick": {"0"}
                                    })
                                    .force_active()
                                ]
                            
                            # Set tempo to 100 BPM for a funky feel
                            e = [ev.intersect(ec().tempo_constraint(100)) for ev in e]
                            
                            # Set tag to funk
                            e = [ev.intersect({"tag": {"funk"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e