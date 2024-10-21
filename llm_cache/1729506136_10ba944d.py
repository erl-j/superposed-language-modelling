# Write a jazzy beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a jazzy beat, we'll use a combination of ride cymbal, hi-hat, kick, and snare,
                            with some ghost notes and occasional crashes. We'll aim for a swing feel and use brushes on the snare.
                            '''
                            e = []
                            # Set tempo to a typical jazz tempo
                            tempo_constraint = ec().tempo_constraint(130)
                            
                            # Ride cymbal pattern (usually on 1, 2, 3, 4 with some variations)
                            for bar in range(4):
                                for beat in range(4):
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"51 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .intersect(tempo_constraint)
                                        .force_active()
                                    ]
                            
                            # Add some ride bell accents
                            e += [ec().intersect({"pitch": {"53 (Drums)"}}).intersect(tempo_constraint).force_active() for _ in range(4)]
                            
                            # Hi-hat (foot) on 2 and 4
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"44 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .intersect(tempo_constraint)
                                        .force_active()
                                    ]
                            
                            # Kick drum (sparse, mainly on 1 and 3, but with variations)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}})
                                    .intersect(tempo_constraint)
                                    .force_active()
                                ]
                            # Add a few more kicks for variation
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).intersect(tempo_constraint).force_active() for _ in range(3)]
                            
                            # Snare (brush) on 2 and 4, with ghost notes
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .intersect(ec().velocity_constraint(80))  # Main snare hits
                                        .intersect(tempo_constraint)
                                        .force_active()
                                    ]
                            # Add ghost notes
                            e += [
                                ec()
                                .intersect({"pitch": {"38 (Drums)"}})
                                .intersect(ec().velocity_constraint(40))  # Ghost notes are quieter
                                .intersect(tempo_constraint)
                                .force_active()
                                for _ in range(8)
                            ]
                            
                            # Add a few crash cymbals for accents
                            e += [ec().intersect({"pitch": {"49 (Drums)"}}).intersect(tempo_constraint).force_active() for _ in range(2)]
                            
                            # Add some tom fills
                            e += [ec().intersect({"pitch": {"45 (Drums)", "47 (Drums)", "48 (Drums)"}}).intersect(tempo_constraint).force_active() for _ in range(4)]
                            
                            # Set tag to jazz
                            e = [ev.intersect({"tag": {"jazz", "-"}}) for ev in e]
                            
                            # Add some optional drum events for more variation
                            e += [ec().intersect({"instrument": {"Drums"}}).intersect(tempo_constraint) for _ in range(10)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e