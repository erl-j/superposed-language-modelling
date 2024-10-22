# add a busy piano line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a busy piano line to the existing beat.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove existing piano notes (if any)
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Add a busy piano line
                            for beat in range(16):  # 16 beats for a 4-bar loop
                                # Add 2-3 notes per beat for a busy feel
                                for _ in range(2):
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "onset/beat": {str(beat)},
                                            "pitch": {str(pitch) for pitch in range(60, 85)},  # Mid to high range
                                        })
                                        .force_active()
                                    ]
                                # Add an optional third note
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(beat)},
                                        "pitch": {str(pitch) for pitch in range(60, 85)},
                                    })
                                ]
                            
                            # Add some off-beat notes for syncopation
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(beat)},
                                        "onset/tick": {"12"},  # Off-beat (halfway between beats)
                                        "pitch": {str(pitch) for pitch in range(60, 85)},
                                    })
                                ]
                            
                            # Add some longer notes for variety
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(bar * 4)},
                                        "offset/beat": {str(bar * 4 + 2)},  # Half-bar duration
                                        "pitch": {str(pitch) for pitch in range(48, 72)},  # Lower range for longer notes
                                    })
                                    .force_active()
                                ]
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e