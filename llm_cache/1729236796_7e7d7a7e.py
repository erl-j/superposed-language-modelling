# drill rap beat, two dotted eight notes followed by an eight note
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a drill rap beat with a rhythm pattern of two dotted eighth notes followed by an eighth note.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set up the basic pattern (two dotted eighth notes followed by an eighth note)
                            pattern = [
                                (0, 0),   # First dotted eighth note
                                (0, 18),  # Second dotted eighth note (1.5 * 12 ticks = 18 ticks)
                                (1, 12)   # Eighth note (on the next beat)
                            ]
                            
                            # Add kicks following the pattern
                            for bar in range(4):
                                for beat in range(4):
                                    for onset_beat, onset_tick in pattern:
                                        e.append(
                                            ec()
                                            .intersect({"pitch": {"36 (Drums)"}})
                                            .intersect({"onset/beat": {str(bar * 4 + beat + onset_beat)}})
                                            .intersect({"onset/tick": {str(onset_tick)}})
                                            .force_active()
                                        )
                            
                            # Add snares on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e.append(
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}})
                                        .intersect({"onset/beat": {str(bar * 4 + beat)}})
                                        .force_active()
                                    )
                            
                            # Add hi-hats (closed)
                            for bar in range(4):
                                for beat in range(4):
                                    for tick in range(0, 24, 6):  # Sixteenth notes
                                        e.append(
                                            ec()
                                            .intersect({"pitch": {"42 (Drums)"}})
                                            .intersect({"onset/beat": {str(bar * 4 + beat)}})
                                            .intersect({"onset/tick": {str(tick)}})
                                            .force_active()
                                        )
                            
                            # Add some open hi-hats for variation
                            for _ in range(8):
                                e.append(ec().intersect({"pitch": {"46 (Drums)"}}).force_active())
                            
                            # Add some additional percussion elements (e.g., toms, cymbals)
                            for _ in range(10):
                                e.append(ec().intersect({"pitch": {"45 (Drums)", "47 (Drums)", "50 (Drums)"}}).force_active())
                            
                            # Set tempo to 140 BPM (typical for drill rap)
                            e = [ev.intersect(ec().tempo_constraint(140)) for ev in e]
                            
                            # Set tag to electronic (closest to drill rap in the given list)
                            e = [ev.intersect({"tag": {"electronic", "-"}}) for ev in e]
                            
                            # Add optional drum notes
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e