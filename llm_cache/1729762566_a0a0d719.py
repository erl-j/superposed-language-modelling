# Create a rhythm using hemiolas across 4 bars
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a rhythm using hemiolas across 4 bars. A hemiola is a rhythmic pattern where two 
                            different time feels are superimposed, typically a feeling of three against two. We'll create this 
                            effect using the drums, emphasizing both a 3/4 and 4/4 feel simultaneously over 4 bars.
                            '''
                            e = []
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Create the 3/4 feel (emphasize every 3rd beat)
                            for bar in range(4):
                                for beat in range(0, 16, 3):  # 0, 3, 6, 9, 12, 15
                                    if bar * 4 + beat < 16:  # Ensure we don't go beyond 4 bars
                                        e += [
                                            ec()
                                            .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + beat)}})
                                            .force_active()
                                        ]

                            # Create the 4/4 feel (emphasize every 4th beat)
                            for bar in range(4):
                                for beat in range(0, 16, 4):  # 0, 4, 8, 12
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + beat)}})
                                        .force_active()
                                    ]

                            # Add hi-hats on every beat for timekeeping
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]

                            # Add some ghost notes for additional rhythm
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"37 (Drums)", "39 (Drums)", "40 (Drums)"}, "velocity": {"40"}})
                                    .force_active()
                                ]

                            # Set tempo (assuming we want to keep the original tempo)
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

                            # Set tag (assuming we want to keep the original tag)
                            e = [ev.intersect({"tag": {tag}}) for ev in e]

                            # Add back other instruments
                            e += e_other

                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]

                            return e