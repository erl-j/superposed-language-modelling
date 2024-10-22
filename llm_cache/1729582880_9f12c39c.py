# Add some short piano chords
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add short piano chords to the existing loop.
                            '''
                            # Keep existing events
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Add piano chords
                            for beat in range(0, 16, 2):  # Add a chord every two beats
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(beat)},
                                        "pitch": {str(pitch) for pitch in range(60, 85)},  # Middle to high range for chords
                                    })
                                    .intersect(ec().velocity_constraint(80))  # Medium-loud velocity
                                    .force_active()
                                    for _ in range(3)  # 3 notes per chord
                                ]
                            
                            # Make the piano chords short
                            for ev in e:
                                if ev.a["instrument"] == {"Piano"}:
                                    ev.intersect({
                                        "offset/beat": {str(int(list(ev.a["onset/beat"])[0]) + 1)},  # End one beat after onset
                                        "offset/tick": {"0"}  # Precisely on the beat
                                    })
                            
                            # Add some optional piano notes for variation
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Piano"},
                                    "pitch": {str(pitch) for pitch in range(60, 85)}
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