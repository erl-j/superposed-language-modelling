# Start over. Start with some sad piano chords.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a sad piano chord progression, we'll use minor chords and slower tempo.
                            We'll create a simple progression: Am - Dm - Em - Am
                            '''
                            e = []
                            # Set a slower tempo for a sad mood
                            tempo = 72
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Define our chord progression
                            chords = [
                                ["A3", "C4", "E4"],  # Am
                                ["D4", "F4", "A4"],  # Dm
                                ["E4", "G4", "B4"],  # Em
                                ["A3", "C4", "E4"],  # Am
                            ]
                            
                            # Place chords on each downbeat
                            for bar in range(4):
                                for note in chords[bar]:
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "pitch": {note},
                                            "onset/beat": {str(bar * 4)},
                                            "offset/beat": {str(bar * 4 + 3)},  # Hold for 3 beats
                                            "velocity": {"60-80"}  # Softer velocity for sad mood
                                        })
                                        .force_active()
                                    )
                            
                            # Add some optional piano notes for variation
                            e += [ec().intersect({"instrument": {"Piano"}}) for _ in range(10)]
                            
                            # Set tag to reflect the mood and style
                            e = [ev.intersect({"tag": {"classical", "romantic", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e