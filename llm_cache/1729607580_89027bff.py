# write a emotional trance piano lead
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create an emotional trance piano lead. This will involve:
                            1. Removing any existing piano notes
                            2. Creating a new piano melody in a higher register
                            3. Setting appropriate velocity and tempo for trance music
                            4. Ensuring the tag reflects the emotional trance style
                            '''
                            # Remove any existing piano notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Create new piano melody
                            for beat in range(16):  # 4 bars of 4 beats each
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(beat)},
                                        "pitch": {str(pitch) for pitch in range(60, 85)},  # Higher register for lead
                                    })
                                    .force_active()
                                ]
                            
                            # Add some shorter notes for variation
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "pitch": {str(pitch) for pitch in range(60, 85)},
                                        "offset/tick": {str(tick) for tick in range(12, 24)}  # Shorter notes
                                    })
                                    .force_active()
                                ]
                            
                            # Set velocity for expressive playing
                            e = [ev.intersect(ec().velocity_constraint(80)) for ev in e]
                            
                            # Set tempo if not already set (typical trance tempo)
                            if tempo is None:
                                e = [ev.intersect(ec().tempo_constraint(138)) for ev in e]
                            else:
                                e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to reflect emotional trance style
                            e = [ev.intersect({"tag": {"trance", "emotional", "piano", "lead"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e