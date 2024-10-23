# write a emotional trance piano lead
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create an emotional trance piano lead. This will involve creating a melodic piano part
                            that fits the trance genre and has an emotional quality.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            # Start over with piano notes
                            e = []

                            # Create a melodic piano lead
                            for bar in range(4):
                                for beat in range(4):
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "onset/beat": {str(beat + bar * 4)},
                                            "pitch": {str(pitch) for pitch in range(60, 85)},  # Mid to high range for emotional lead
                                        })
                                        .force_active()
                                    ]

                            # Add some longer notes for emotional effect
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Piano"},
                                    "onset/beat": {"0", "8"},  # Start of first and third bar
                                    "offset/beat": {"3", "11"},  # Hold for 3 beats
                                    "pitch": {str(pitch) for pitch in range(72, 85)},  # Higher notes for emphasis
                                })
                                .force_active()
                                for _ in range(2)
                            ]

                            # Add some arpeggios for trance feel
                            for bar in [1, 3]:  # Second and fourth bar
                                for tick in range(0, 24, 6):  # Every 6th tick (16th notes)
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "onset/beat": {str(bar * 4)},
                                            "onset/tick": {str(tick)},
                                            "pitch": {str(pitch) for pitch in range(60, 85)},
                                        })
                                        .force_active()
                                    ]

                            # Set velocity for dynamic expression
                            e = [ev.intersect(ec().velocity_constraint(80)) for ev in e]  # Medium-high velocity for emotional impact

                            # Preserve the original tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

                            # Set tags
                            e = [ev.intersect({"tag": {"trance", "emotional", "piano", "lead"}}) for ev in e]

                            # Add back other instruments
                            e += e_other

                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]

                            return e