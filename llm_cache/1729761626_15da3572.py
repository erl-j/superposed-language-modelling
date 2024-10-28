# Create a bassline with syncopation
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a syncopated bassline that adds groove to the existing beat.
                            Syncopation involves emphasizing off-beats and weak beats, which will create tension and interest.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Bass"})]
                            # Start over with bass notes
                            e = []

                            # Create syncopated bass pattern
                            # We'll place bass notes on off-beats and some weak beats
                            syncopated_beats = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
                            syncopated_ticks = ["12", "18"] # These represent off-beat positions

                            for beat in syncopated_beats:
                                for tick in syncopated_ticks:
                                    e += [
                                        ec()
                                        .intersect(
                                            {
                                                "instrument": {"Bass"},
                                                "onset/beat": {beat},
                                                "onset/tick": {tick},
                                                "pitch": {str(pitch) for pitch in range(30, 55)}, # Bass range
                                            }
                                        )
                                    ]

                            # Add some longer notes for variety
                            e += [
                                ec()
                                .intersect({"instrument": {"Bass"}, "offset/beat": {"2", "3", "4"}})
                                .force_active()
                                for _ in range(4)
                            ]

                            # Add some short staccato notes for funk feel
                            e += [
                                ec()
                                .intersect({"instrument": {"Bass"}, "offset/tick": {"6", "12"}})
                                .force_active()
                                for _ in range(6)
                            ]

                            # Ensure at least 10 bass notes are active
                            for _ in range(10):
                                e += [ec().intersect({"instrument": {"Bass"}}).force_active()]

                            # Add up to 10 more optional bass notes for variety
                            e += [ec().intersect({"instrument": {"Bass"}}) for _ in range(10)]

                            # Intersect with current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

                            # Set tag to current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]

                            # Add back other instruments
                            e += e_other

                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]

                            return e