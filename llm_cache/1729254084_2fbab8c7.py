# Give me a classic reggaeton beat over 4 bars
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a classic reggaeton beat over 4 bars, we'll use the characteristic "Dembow" rhythm.
                            This includes a kick drum, snare/clap, and hi-hats, with the iconic 3-3-2 rhythm.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Kick drum pattern (36)
                            kick_pattern = [0, 0, 2, 0, 2, 3, 0]
                            for bar in range(4):
                                for i, beat in enumerate(kick_pattern):
                                    if beat > 0:
                                        e.append(
                                            ec()
                                            .intersect({"pitch": {"36 (Drums)"}})
                                            .intersect({"onset/beat": {str(i + bar * 4)}})
                                            .force_active()
                                        )
                            
                            # Snare/Clap pattern (38 or 39)
                            snare_pattern = [0, 0, 0, 2, 0, 0, 2]
                            for bar in range(4):
                                for i, beat in enumerate(snare_pattern):
                                    if beat > 0:
                                        e.append(
                                            ec()
                                            .intersect({"pitch": {"38 (Drums)", "39 (Drums)"}})
                                            .intersect({"onset/beat": {str(i + bar * 4)}})
                                            .force_active()
                                        )
                            
                            # Hi-hat pattern (42 for closed, 46 for open)
                            hihat_pattern = [2, 0, 2, 0, 2, 0, 2]
                            for bar in range(4):
                                for i, beat in enumerate(hihat_pattern):
                                    if beat > 0:
                                        e.append(
                                            ec()
                                            .intersect({"pitch": {"42 (Drums)", "46 (Drums)"}})
                                            .intersect({"onset/beat": {str(i + bar * 4)}})
                                            .force_active()
                                        )
                            
                            # Add some optional percussion hits
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo to typical reggaeton tempo (around 90-100 BPM)
                            e = [ev.intersect(ec().tempo_constraint(95)) for ev in e]
                            
                            # Set tag to reggae (closest to reggaeton in the available tags)
                            e = [ev.intersect({"tag": {"reggae", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e