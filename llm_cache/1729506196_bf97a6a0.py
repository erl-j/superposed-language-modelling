# Write a modern jazz beat with syncopation and polyrythms
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a modern jazz beat with syncopation and polyrhythms, we'll use a combination of different drum sounds,
                            off-beat accents, and overlapping rhythmic patterns.
                            '''
                            e = []
                            # Clear existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set up a 7/4 time signature for added complexity
                            bar_length = 7
                            
                            # Ride cymbal pattern (continuous eighths with accents)
                            for bar in range(4):
                                for beat in range(bar_length):
                                    for eighth in [0, 12]:
                                        accent = 1 if beat % 2 == 0 else 0.7
                                        e.append(
                                            ec()
                                            .intersect({"pitch": {"51 (Drums)"}, "onset/beat": {str(bar * bar_length + beat)}, "onset/tick": {str(eighth)}})
                                            .intersect(ec().velocity_constraint(int(accent * 100)))
                                            .force_active()
                                        )
                            
                            # Kick drum pattern (syncopated)
                            kick_pattern = [0, 12, 36, 60, 84, 108, 132, 156, 180, 204]
                            for tick in kick_pattern:
                                e.append(
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/tick": {str(tick % 24)}})
                                    .intersect(ec().velocity_constraint(110))
                                    .force_active()
                                )
                            
                            # Snare drum pattern (off-beat accents)
                            snare_pattern = [48, 120, 168, 216]
                            for tick in snare_pattern:
                                e.append(
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}, "onset/tick": {str(tick % 24)}})
                                    .intersect(ec().velocity_constraint(95))
                                    .force_active()
                                )
                            
                            # Hi-hat pattern (polyrhythm: 3 against 4)
                            for bar in range(4):
                                for beat in range(bar_length):
                                    for third in [0, 8, 16]:
                                        e.append(
                                            ec()
                                            .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(bar * bar_length + beat)}, "onset/tick": {str(third)}})
                                            .intersect(ec().velocity_constraint(80))
                                            .force_active()
                                        )
                            
                            # Tom fills (occasional, to add variety)
                            tom_pattern = [45, 47, 50]
                            for i in range(3):
                                e.append(
                                    ec()
                                    .intersect({"pitch": {f"{tom_pattern[i]} (Drums)"}, "onset/beat": {str(26 + i)}})
                                    .intersect(ec().velocity_constraint(90))
                                    .force_active()
                                )
                            
                            # Set tempo to a moderate jazz tempo
                            e = [ev.intersect(ec().tempo_constraint(132)) for ev in e]
                            
                            # Set tag to jazz
                            e = [ev.intersect({"tag": {"jazz", "modern"}}) for ev in e]
                            
                            # Add some optional percussion hits for variety
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e