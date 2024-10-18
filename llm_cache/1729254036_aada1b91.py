# Give me a classic reggaeton beat.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a classic reggaeton beat, we'll focus on the characteristic "Dem Bow" rhythm.
                            This includes a kick drum pattern, snare/clap, and the iconic reggaeton percussion sounds.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Kick drum pattern (36)
                            kick_pattern = [0, 0, 2, 3, 4, 4, 6, 7]
                            for beat in kick_pattern:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Snare/Clap pattern (38) - on beats 2 and 4
                            for beat in [2, 6]:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Hi-hat pattern (42) - steady eighth notes
                            for beat in range(8):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"0"}})
                                    .force_active()
                                ]
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
                            
                            # Reggaeton percussion (conga-like sound) (63)
                            conga_pattern = [0, 1, 3, 4, 6, 7]
                            for beat in conga_pattern:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"63 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Add some variation with low tom (41) on the last beat
                            e += [
                                ec()
                                .intersect({"pitch": {"41 (Drums)"}, "onset/beat": {"7"}, "onset/tick": {"12"}})
                                .force_active()
                            ]
                            
                            # Set tempo to 90-100 BPM (typical for reggaeton)
                            e = [ev.intersect(ec().tempo_constraint(95)) for ev in e]
                            
                            # Set tag to reggae (closest to reggaeton in the given list)
                            e = [ev.intersect({"tag": {"reggae", "-"}}) for ev in e]
                            
                            # Add up to 10 optional percussion hits for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e  # return the events