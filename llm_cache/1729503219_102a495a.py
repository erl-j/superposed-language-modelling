# Write some sad piano chords.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a sad piano chord progression, we'll use minor chords and slower tempo.
                            We'll create a sequence of Am - F - C - Em chords, which is a common sad progression.
                            '''
                            # Remove any existing piano notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Define our chord progression
                            chords = [
                                [57, 60, 64],  # Am (A3, C4, E4)
                                [53, 57, 60],  # F  (F3, A3, C4)
                                [48, 52, 55],  # C  (C3, E3, G3)
                                [52, 55, 59]   # Em (E3, G3, B3)
                            ]
                            
                            # Place chords on each beat
                            for bar in range(4):
                                for beat in range(4):
                                    chord = chords[bar]
                                    for note in chord:
                                        e.append(
                                            ec()
                                            .intersect({
                                                "instrument": {"Piano"},
                                                "pitch": {str(note)},
                                                "onset/beat": {str(bar * 4 + beat)},
                                                "offset/beat": {str(bar * 4 + beat + 1)},  # Hold for one beat
                                                "velocity": {"60"}  # Soft velocity for sad mood
                                            })
                                            .force_active()
                                        )
            
                            # Set a slower tempo for a sadder feel
                            e = [ev.intersect(ec().tempo_constraint(72)) for ev in e]
                            
                            # Set tag to classical or romantic for a more melancholic feel
                            e = [ev.intersect({"tag": {"classical", "romantic"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e