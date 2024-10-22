# add some uptempo dancy 909 drums
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add uptempo dancy 909 drums to the existing beat.
                            The Roland TR-909 is known for its distinctive sounds, so we'll focus on its key elements.
                            '''
                            # Preserve existing events
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove existing drums to replace with 909 style
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set a faster tempo for a dance feel (if not already uptempo)
                            new_tempo = 128  # A common tempo for dance music
                            e = [ev.intersect(ec().tempo_constraint(new_tempo)) for ev in e]
                            
                            # Add 909-style kick drum (usually on every quarter note)
                            for beat in range(16):
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                    .intersect(ec().velocity_constraint(100))  # Strong kick
                                    .force_active()
                                )
                            
                            # Add 909-style snare (typically on 2 and 4)
                            for beat in [1, 3, 5, 7, 9, 11, 13, 15]:
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/beat": {str(beat)}})
                                    .intersect(ec().velocity_constraint(90))
                                    .force_active()
                                )
                            
                            # Add 909-style closed hi-hats (on every 8th note)
                            for beat in range(16):
                                for tick in [0, 12]:
                                    e.append(
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {str(tick)}})
                                        .intersect(ec().velocity_constraint(80))
                                        .force_active()
                                    )
                            
                            # Add 909-style open hi-hats (occasional)
                            for beat in [2, 6, 10, 14]:
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"46 (Drums)"}, "onset/beat": {str(beat)}})
                                    .intersect(ec().velocity_constraint(85))
                                    .force_active()
                                )
                            
                            # Add some 909-style toms for fill (in the last bar)
                            for beat in [12, 13, 14, 15]:
                                e.append(
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"45 (Drums)", "47 (Drums)", "48 (Drums)"}, "onset/beat": {str(beat)}})
                                    .intersect(ec().velocity_constraint(75))
                                    .force_active()
                                )
                            
                            # Add up to 10 optional percussion hits for variety
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Preserve the original tag, but add 'dance-eletric' if not present
                            original_tag = next(iter(e[0].a["tag"]))
                            new_tag = {"dance-eletric", original_tag} if original_tag != "dance-eletric" else {original_tag}
                            e = [ev.intersect({"tag": new_tag}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e