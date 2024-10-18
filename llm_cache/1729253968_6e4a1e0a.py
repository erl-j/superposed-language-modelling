# Give me a reggaeton beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a reggaeton beat, we'll focus on the characteristic "dembow" rhythm, 
                            which typically includes kicks, snares, and a distinctive percussion pattern.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})}]
                            
                            # Kick drum pattern (typically on beats 1 and 2.5)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}}).force_active(),
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}, "onset/tick": {"12"}}).force_active()
                                ]
                            
                            # Snare/Clap (typically on beats 2 and 4)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + 1)}}).force_active(),
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + 3)}}).force_active()
                                ]
                            
                            # Dembow rhythm (usually played on timbales or similar percussion)
                            dembow_pattern = [0, 12, 24, 36, 48, 60, 72, 84]
                            for bar in range(4):
                                for tick in dembow_pattern:
                                    e += [
                                        ec().intersect({"pitch": {"47 (Drums)"}, "onset/beat": {str(bar * 4)}, "onset/tick": {str(tick)}}).force_active()
                                    ]
                            
                            # Hi-hats (usually steady eighth notes)
                            for bar in range(4):
                                for beat in range(4):
                                    e += [
                                        ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(bar * 4 + beat)}}).force_active(),
                                        ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(bar * 4 + beat)}, "onset/tick": {"12"}}).force_active()
                                    ]
                            
                            # Add some optional percussion hits
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set tempo to typical reggaeton tempo (around 90-100 BPM)
                            e = [ev.intersect(ec().tempo_constraint(95)) for ev in e]
                            
                            # Set tag to latin (closest to reggaeton in the given list)
                            e = [ev.intersect({"tag": {"latin", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e