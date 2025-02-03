# start with some piano chords
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a beat starting with some piano chords. We'll set up a simple chord progression
                            with four chords, one per bar, and leave room for other instruments to be added later.
                            '''
                            e = []
                            # Set up piano chords
                            for bar in range(4):
                                chord_onset = bar * 96  # 96 ticks per bar (24 ticks per beat * 4 beats)
                                # Create a triad chord (3 notes)
                                for _ in range(3):
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Piano"},
                                            "onset/global_tick": {str(chord_onset)},
                                            "duration": {"1"},  # whole note duration
                                            "pitch": {str(pitch) for pitch in range(60, 85)}  # middle range of piano
                                        })
                                        .force_active()
                                    ]

                            # Set tempo (let's use a moderate tempo)
                            e = [ev.intersect(ec().tempo_constraint(120)) for ev in e]

                            # Set a general tag (since we don't have a specific genre yet)
                            e = [ev.intersect({"tag": {"pop", "-"}}) for ev in e]

                            # Add some optional piano notes for potential embellishments
                            e += [ec().intersect({"instrument": {"Piano"}}) for _ in range(10)]

                            # Leave room for other instruments to be added later
                            # (drums, bass, etc. could be added in subsequent constraints)

                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]

                            return e