# Add some more bass notes
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add more bass notes to the existing beat, enhancing the bassline while keeping the original structure.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside non-bass instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Find existing bass notes
                            existing_bass = [ev for ev in e if "Bass" in ev.a["instrument"]]
                            
                            # Start with existing bass notes
                            new_bass = existing_bass.copy()
                            
                            # Add up to 8 more bass notes
                            for _ in range(8):
                                new_bass.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in range(30, 55)},  # Wider pitch range for variety
                                    })
                                    .force_active()
                                )
                            
                            # Add some off-beat bass notes for groove
                            for beat in range(16):
                                if beat % 2 != 0:  # Off-beats
                                    new_bass.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "onset/beat": {str(beat)},
                                            "pitch": {str(pitch) for pitch in range(35, 50)},
                                        })
                                        .force_active()
                                    )
                            
                            # Ensure consistent tempo and tag
                            new_bass = [ev.intersect(ec().tempo_constraint(tempo)) for ev in new_bass]
                            new_bass = [ev.intersect({"tag": {tag}}) for ev in new_bass]
                            
                            # Combine new bass notes with other instruments
                            e = new_bass + e_other
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e