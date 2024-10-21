# Now make a syncopated piano part
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a syncopated piano part to the existing funk beat and bassline.
                            Syncopation in funk often emphasizes the off-beats and uses rhythmic displacement.
                            '''
                            # Remove inactive notes and any existing piano notes
                            e = [ev for ev in e if ev.is_active() and ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Define syncopated rhythms (emphasizing off-beats)
                            syncopated_beats = [
                                {"onset/beat": {"0"}, "onset/tick": {"12"}},
                                {"onset/beat": {"1"}, "onset/tick": {"12"}},
                                {"onset/beat": {"2"}, "onset/tick": {"12"}},
                                {"onset/beat": {"3"}, "onset/tick": {"12"}},
                                {"onset/beat": {"0", "1", "2", "3"}, "onset/tick": {"6", "18"}},
                            ]
                            
                            # Add syncopated piano chords
                            for rhythm in syncopated_beats:
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Piano"}})
                                    .intersect(rhythm)
                                    .intersect({"pitch": {str(pitch) for pitch in range(60, 85)}})  # Mid to high range for piano chords
                                    .force_active()
                                ]
                            
                            # Add some optional piano notes for variation
                            e += [
                                ec()
                                .intersect({"instrument": {"Piano"}})
                                .intersect({"pitch": {str(pitch) for pitch in range(50, 85)}})  # Wider range for optional notes
                                for _ in range(10)
                            ]
                            
                            # Add some staccato notes (short duration) for funkiness
                            e += [
                                ec()
                                .intersect({"instrument": {"Piano"}})
                                .intersect({"pitch": {str(pitch) for pitch in range(50, 85)}})
                                .intersect(ec().offset_constraint(0, 6))  # Short duration (less than a quarter beat)
                                .force_active()
                                for _ in range(5)
                            ]
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to reach n_events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e