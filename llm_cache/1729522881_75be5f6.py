# add a bass line
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a funky bass line that complements the existing drum pattern.
                            The bass line will have a mix of sustained notes and short, punchy notes.
                            '''
                            # Remove any existing bass notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define bass pitches for a funky line (E, G, A, C)
                            bass_pitches = {40, 43, 45, 48}
                            
                            # Add a bass note on beat 1 of each bar
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": bass_pitches,
                                        "onset/beat": {str(bar * 4)},
                                        "offset/beat": {str(bar * 4 + 2)},  # Sustain for 2 beats
                                    })
                                    .intersect(ec().velocity_constraint(100))  # Strong velocity for emphasis
                                    .force_active()
                                ]
                            
                            # Add shorter bass notes on off-beats
                            for bar in range(4):
                                for beat in [0.5, 1.5, 2.5, 3.5]:
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"},
                                            "pitch": bass_pitches,
                                            "onset/beat": {str(int(bar * 4 + beat))},
                                            "onset/tick": {"12"},  # Half-beat offset
                                            "offset/beat": {str(int(bar * 4 + beat + 0.5))},  # Short duration
                                        })
                                        .intersect(ec().velocity_constraint(80))  # Slightly softer
                                        .force_active()
                                    ]
                            
                            # Add some optional bass notes for variation
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Bass"},
                                    "pitch": bass_pitches,
                                })
                                for _ in range(8)
                            ]
                            
                            # Set tempo to match existing loop
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to funk
                            e = [ev.intersect({"tag": {"funk"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e