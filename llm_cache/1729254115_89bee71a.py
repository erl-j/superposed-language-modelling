# Give me a classic reggaeton beat over 4 bars at 120 bpm
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a classic reggaeton beat over 4 bars at 120 bpm, we'll use the characteristic "Dembow" rhythm.
                            This includes a kick drum, snare/clap, and a distinctive hi-hat pattern.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Kick drum pattern (36): on the 1 and between 3 and 4 of each bar
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}}).force_active(),
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}, "onset/tick": {"12"}}).force_active()
                                ]
                            
                            # Snare/Clap pattern (38 or 39): on the 2 and 4 of each bar
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"38 (Drums)", "39 (Drums)"}, "onset/beat": {str(bar * 4 + 1)}}).force_active(),
                                    ec().intersect({"pitch": {"38 (Drums)", "39 (Drums)"}, "onset/beat": {str(bar * 4 + 3)}}).force_active()
                                ]
                            
                            # Hi-hat pattern (42): continuous eighth notes
                            for bar in range(4):
                                for beat in range(4):
                                    e += [
                                        ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(bar * 4 + beat)}, "onset/tick": {"0"}}).force_active(),
                                        ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(bar * 4 + beat)}, "onset/tick": {"12"}}).force_active()
                                    ]
                            
                            # Add some variation with open hi-hats (46) and additional percussion (e.g., 47 for low tom)
                            e += [ec().intersect({"pitch": {"46 (Drums)", "47 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Set tempo to 120 bpm
                            e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]
                            
                            # Set tag to latin (closest to reggaeton in the available tags)
                            e = [ev.intersect({"tag": {"latin", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e