# Now make a syncopated piano part
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a syncopated piano part that adds to the existing rhythm.
                            Syncopation is achieved by placing notes on off-beats and weak beats.
                            '''
                            # Remove inactive notes and any existing piano notes
                            e = [ev for ev in e if ev.is_active() and ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Create syncopated piano notes
                            for bar in range(4):
                                # Add notes on the "and" of beats 1 and 3
                                for beat in [0, 2]:
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "onset/beat": {str(bar * 4 + beat)},
                                            "onset/tick": {"12"}, # 12 ticks is halfway between beats
                                            "pitch": {str(pitch) for pitch in range(60, 85)}, # mid to high range
                                        })
                                        .force_active()
                                    )
                                
                                # Add a note on the "e" of beat 2 or 4 (alternating)
                                e.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(bar * 4 + (1 if bar % 2 == 0 else 3))},
                                        "onset/tick": {"18"}, # 18 ticks is 3/4 of the way between beats
                                        "pitch": {str(pitch) for pitch in range(60, 85)},
                                    })
                                    .force_active()
                                )
            
            
                            # Add some optional piano notes for variety
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Piano"},
                                    "pitch": {str(pitch) for pitch in range(60, 85)},
                                })
                                for _ in range(8)
                            ]
            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e