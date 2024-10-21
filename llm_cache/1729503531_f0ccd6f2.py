# Add some chords to go with the current melody
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a chord progression that complements the existing melody and bassline,
                            we'll add piano chords on the first beat of each bar, with some additional chord hits.
                            We'll use common funk chord voicings and ensure they don't clash with the existing notes.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove any existing piano chords
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Find existing melody notes
                            melody_notes = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums", "Bass"})]
                            
                            # Define chord voicings (these are just examples, adjust as needed)
                            chord_voicings = [
                                [0, 4, 7, 10],  # 7th chord
                                [0, 4, 7, 11],  # maj7 chord
                                [0, 3, 7, 10],  # min7 chord
                                [0, 4, 7, 9],   # 6th chord
                            ]
                            
                            # Add a chord on the first beat of each bar
                            for bar in range(4):
                                chord_root = ec().intersect({"instrument": {"Piano"}, "onset/beat": {str(bar * 4)}})
                                e.append(chord_root.force_active())
                                
                                # Add the chord voicing
                                voicing = random.choice(chord_voicings)
                                for interval in voicing[1:]:  # Skip the root note as it's already added
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "onset/beat": {str(bar * 4)},
                                            "pitch": {f"+{interval}"}  # Relative to the root note
                                        })
                                        .force_active()
                                    )
                            
                            # Add some additional chord hits (up to 8)
                            for _ in range(8):
                                e.append(ec().intersect({"instrument": {"Piano"}}))
                            
                            # Ensure chords don't clash with melody
                            for chord_note in [ev for ev in e if ev.a["instrument"] == {"Piano"}]:
                                for melody_note in melody_notes:
                                    if (chord_note.a["onset/beat"] == melody_note.a["onset/beat"] and
                                        chord_note.a["onset/tick"] == melody_note.a["onset/tick"]):
                                        # Adjust the chord note to avoid clash
                                        chord_note = chord_note.intersect(ec().pitch_constraint(lambda p: abs(p - melody_note.a["pitch"]) > 1))
                            
                            # Set tempo and tag (assuming we're keeping the funk style)
                            e = [ev.intersect(ec().tempo_constraint(96)) for ev in e]
                            e = [ev.intersect({"tag": {"funk", "-"}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e