# Start with some sad piano chords
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a sad piano chord progression, we'll use minor chords and a slow tempo.
                            We'll create a simple progression: Am - Dm - Em - Am
                            '''
                            e = []
                            # Set a slow tempo for a sad mood
                            tempo = 70
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Define our chord progression
                            chords = [
                                [57, 60, 64],  # Am
                                [62, 65, 69],  # Dm
                                [64, 67, 71],  # Em
                                [57, 60, 64]   # Am
                            ]
                            
                            # Place chords on each downbeat
                            for bar in range(4):
                                for note in chords[bar]:
                                    e.append(
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "pitch": {str(note)},
                                            "onset/beat": {str(bar * 4)},
                                            "offset/beat": {str(bar * 4 + 3)},  # Hold for 3 beats
                                            "velocity": {"60", "70", "80"}  # Softer velocity for sad mood
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