# Create a jazzy drum beat with syncopation
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a jazzy drum beat with syncopation. This will include ride cymbal, kick, snare, and hi-hat,
                            with emphasis on off-beats and some ghost notes for added complexity.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add ride cymbal pattern (typically on quarter notes, but with some variations)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"51 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
            
                            # Add kick drum (on 1 and 3, with some syncopation)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}})
                                    .force_active()
                                ]
                                # Add syncopated kick
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
            
                            # Add snare (on 2 and 4, with some ghost notes)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}})
                                    .force_active()
                                ]
                                # Add ghost note
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + 3)}, "onset/tick": {"12"}})
                                    .intersect(ec().velocity_constraint(30))
                                    .force_active()
                                ]
            
                            # Add hi-hat (on off-beats for syncopation)
                            for beat in range(16):
                                if beat % 2 != 0:  # off-beats
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}})
                                        .force_active()
                                    ]
            
                            # Add some optional drum hits for variety
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
            
                            # Set tempo (jazz often has a medium tempo, around 120-140 BPM)
                            e = [ev.intersect(ec().tempo_constraint(130)) for ev in e]
            
                            # Set tag to jazz
                            e = [ev.intersect({"tag": {"jazz"}}) for ev in e]
            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e