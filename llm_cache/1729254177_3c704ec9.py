# Give me a classic reggaeton beat over 4 bars at 120 bpm. make sure to include the dembow rythm.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a classic reggaeton beat with the dembow rhythm over 4 bars at 120 bpm.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Define the dembow rhythm pattern (over 2 beats)
                            dembow_pattern = [
                                (0, 0), (0, 12),  # Kick
                                (1, 0),           # Snare
                                (1, 12)           # Kick
                            ]
                            
                            # Add the dembow rhythm for 4 bars (8 repetitions)
                            for bar in range(4):
                                for pattern_repeat in range(2):
                                    for beat, tick in dembow_pattern:
                                        onset_beat = bar * 4 + pattern_repeat * 2 + beat
                                        if beat == 0 or beat == 1 and tick == 12:  # Kick
                                            e.append(
                                                ec()
                                                .intersect({"pitch": {"36 (Drums)"}})
                                                .intersect({"onset/beat": {str(onset_beat)}, "onset/tick": {str(tick)}})
                                                .force_active()
                                            )
                                        elif beat == 1 and tick == 0:  # Snare
                                            e.append(
                                                ec()
                                                .intersect({"pitch": {"38 (Drums)"}})
                                                .intersect({"onset/beat": {str(onset_beat)}, "onset/tick": {str(tick)}})
                                                .force_active()
                                            )
                            
                            # Add hi-hats on every beat
                            for beat in range(16):
                                e.append(
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}})
                                    .intersect({"onset/beat": {str(beat)}})
                                    .force_active()
                                )
                            
                            # Add some additional percussion elements
                            # Clave-like sound on the "and" of 2 and 4 in each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e.append(
                                        ec()
                                        .intersect({"pitch": {"75 (Drums)"}})  # Clave or similar percussion
                                        .intersect({"onset/beat": {str(bar * 4 + beat)}, "onset/tick": {"12"}})
                                        .force_active()
                                    )
                            
                            # Add up to 10 optional percussion hits
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo to 120 bpm
                            e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]
                            
                            # Set tag to latin (closest to reggaeton in the given list)
                            e = [ev.intersect({"tag": {"latin", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events