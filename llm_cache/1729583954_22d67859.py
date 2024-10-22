# Add some piano melody to the chords
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a piano melody that complements the existing chords.
                            We'll keep the existing notes and add new piano notes for the melody.
                            '''
                            # Keep all existing active notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Find existing piano chords
                            piano_chords = [ev for ev in e if ev.a["instrument"].intersection({"Piano"})]
                            
                            # Add melodic piano notes
                            for _ in range(16):  # Add up to 16 melodic notes
                                new_note = ec().intersect({
                                    "instrument": {"Piano"},
                                    "pitch": {str(pitch) for pitch in range(60, 85)},  # Higher range for melody
                                })
                                # Ensure the new note doesn't overlap with existing chords
                                for chord in piano_chords:
                                    new_note = new_note.intersect({
                                        "onset/beat": {str(beat) for beat in range(16) if beat not in chord.a["onset/beat"]}
                                    })
                                e.append(new_note.force_active())
                            
                            # Add some shorter notes for more melodic variation
                            for _ in range(8):
                                e.append(ec().intersect({
                                    "instrument": {"Piano"},
                                    "pitch": {str(pitch) for pitch in range(60, 85)},
                                    "offset/tick": {str(tick) for tick in range(12)}  # Shorter duration
                                }).force_active())
                            
                            # Maintain the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Maintain the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add some velocity variation for expressiveness
                            e = [ev.intersect(ec().velocity_constraint(80 + i % 20)) for i, ev in enumerate(e) if ev.a["instrument"].intersection({"Piano"})]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e