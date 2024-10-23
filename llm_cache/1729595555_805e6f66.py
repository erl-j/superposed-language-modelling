# add some claps and toms on offbeats
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add claps and toms on offbeats to the existing beat.
                            Claps are typically on MIDI note 39, and toms can be 45, 47, or 50 for low, mid, and high toms respectively.
                            '''
                            # Keep existing events
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Add claps on offbeats (beats 2 and 4 of each bar)
                            for bar in range(4):
                                for beat in [1, 3]:  # 2nd and 4th beat of each bar
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Drums"},
                                            "pitch": {"39 (Drums)"},  # Clap
                                            "onset/beat": {str(beat + bar * 4)},
                                            "onset/tick": {"12"}  # Slightly off-beat (half a beat)
                                        })
                                        .force_active()
                                    ]
                            
                            # Add toms on other offbeats
                            tom_pitches = ["45 (Drums)", "47 (Drums)", "50 (Drums)"]  # Low, mid, high toms
                            for bar in range(4):
                                for beat in range(4):
                                    if beat % 2 == 0:  # On 1st and 3rd beats of each bar
                                        e += [
                                            ec()
                                            .intersect({
                                                "instrument": {"Drums"},
                                                "pitch": set(tom_pitches),  # Any of the tom pitches
                                                "onset/beat": {str(beat + bar * 4)},
                                                "onset/tick": {"12"}  # Slightly off-beat (half a beat)
                                            })
                                            .force_active()
                                        ]
                            
                            # Add some optional tom fills
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Drums"},
                                    "pitch": set(tom_pitches)
                                }) 
                                for _ in range(10)
                            ]
                            
                            # Preserve tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e