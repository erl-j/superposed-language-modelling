# make a more interesting hi-hat pattern please
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a more interesting hi-hat pattern while maintaining the funk groove.
                            We'll use a combination of closed and open hi-hats, with some syncopation and ghost notes.
                            '''
                            # Remove existing hi-hats
                            e = [ev for ev in e if ev.a["pitch"].isdisjoint({"42 (Drums)", "46 (Drums)"})]
                            
                            # Create a basic pattern for closed hi-hats (every 8th note)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"0"}})
                                    .intersect(ec().velocity_constraint(80))  # Normal velocity
                                    .force_active()
                                ]
                            
                            # Add some syncopated closed hi-hats
                            syncopated_beats = [0, 2, 6, 10, 14]
                            for beat in syncopated_beats:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"12"}})
                                    .intersect(ec().velocity_constraint(70))  # Slightly lower velocity
                                    .force_active()
                                ]
                            
                            # Add open hi-hats on some offbeats
                            open_hihat_beats = [1, 5, 9, 13]
                            for beat in open_hihat_beats:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"46 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"12"}})
                                    .intersect(ec().velocity_constraint(90))  # Slightly higher velocity
                                    .force_active()
                                ]
                            
                            # Add some ghost notes (very quiet closed hi-hats)
                            ghost_beats = [3, 7, 11, 15]
                            for beat in ghost_beats:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"18"}})
                                    .intersect(ec().velocity_constraint(30))  # Very low velocity
                                    .force_active()
                                ]
                            
                            # Add some flexibility for variation
                            e += [ec().intersect({"pitch": {"42 (Drums)", "46 (Drums)"}}) for _ in range(10)]
                            
                            # Preserve tempo and tag
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e