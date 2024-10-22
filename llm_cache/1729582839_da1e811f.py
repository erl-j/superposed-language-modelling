# Add some piano stabs
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add piano stabs to the existing loop, keeping the current rhythm and adding some accents.
                            '''
                            # Keep existing events
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove any existing piano notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Add piano stabs
                            for beat in range(16):  # 4 bars of 4 beats each
                                if beat % 4 == 0:  # On the 1 of each bar
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "onset/beat": {str(beat)},
                                            "pitch": {str(pitch) for pitch in range(60, 85)},  # Mid to high range for stabs
                                            "offset/beat": {str(beat + 1)}  # Short duration
                                        })
                                        .intersect(ec().velocity_constraint(100))  # Louder for emphasis
                                        .force_active()
                                    )
                                elif beat % 2 == 0:  # On even beats (2 and 4 of each bar)
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "onset/beat": {str(beat)},
                                            "pitch": {str(pitch) for pitch in range(60, 85)},
                                            "offset/beat": {str(beat + 1)}
                                        })
                                        .intersect(ec().velocity_constraint(80))  # Slightly softer
                                        .force_active()
                                    )
                            
                            # Add some optional syncopated stabs
                            for _ in range(5):
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(beat) for beat in range(16)},
                                        "onset/tick": {"12", "18"},  # Syncopated timing
                                        "pitch": {str(pitch) for pitch in range(60, 85)},
                                        "offset/beat": {str(beat) for beat in range(1, 17)}
                                    })
                                )
                            
                            # Preserve tempo and tag
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e