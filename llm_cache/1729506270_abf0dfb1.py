# Write a modern jazz beat with toms and rides
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a modern jazz beat with toms and rides, we'll create a complex rhythm
                            with syncopation, varied dynamics, and some polyrhythmic elements.
                            '''
                            e = []
                            # Clear all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set up ride cymbal pattern (traditional jazz ride pattern with variations)
                            for bar in range(4):
                                for beat in range(4):
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"51 (Drums)"}, "onset/beat": {str(bar*4 + beat)}, "onset/tick": {"0"}})
                                        .intersect(ec().velocity_constraint(90))
                                        .force_active()
                                    ]
                                    # Add some ghost notes on the ride
                                    if beat % 2 == 1:
                                        e += [
                                            ec()
                                            .intersect({"pitch": {"51 (Drums)"}, "onset/beat": {str(bar*4 + beat)}, "onset/tick": {"12"}})
                                            .intersect(ec().velocity_constraint(60))
                                            .force_active()
                                        ]
                            
                            # Add hi-hat foot chick on 2 and 4
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"44 (Drums)"}, "onset/beat": {str(bar*4 + beat)}})
                                        .intersect(ec().velocity_constraint(70))
                                        .force_active()
                                    ]
                            
                            # Add syncopated snare hits
                            syncopated_snare = [2, 6, 10, 13]
                            for beat in syncopated_snare:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(beat)}})
                                    .intersect(ec().velocity_constraint(85))
                                    .force_active()
                                ]
                            
                            # Add tom fills
                            tom_pitches = ["45 (Drums)", "47 (Drums)", "50 (Drums)"]  # Low, mid, high toms
                            for bar in [1, 3]:
                                for i, tom in enumerate(tom_pitches):
                                    e += [
                                        ec()
                                        .intersect({"pitch": {tom}, "onset/beat": {str(bar*4 + 3)}, "onset/tick": {str(i*8)}})
                                        .intersect(ec().velocity_constraint(80))
                                        .force_active()
                                    ]
                            
                            # Add some kick drum hits
                            kick_pattern = [0, 4, 8, 10, 12]
                            for beat in kick_pattern:
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                    .intersect(ec().velocity_constraint(95))
                                    .force_active()
                                ]
                            
                            # Add some crash cymbal accents
                            e += [
                                ec()
                                .intersect({"pitch": {"49 (Drums)"}, "onset/beat": {"0"}})
                                .intersect(ec().velocity_constraint(100))
                                .force_active()
                            ]
                            e += [
                                ec()
                                .intersect({"pitch": {"49 (Drums)"}, "onset/beat": {"12"}})
                                .intersect(ec().velocity_constraint(90))
                                .force_active()
                            ]
                            
                            # Set tempo to 160 BPM (a common tempo for modern jazz)
                            e = [ev.intersect(ec().tempo_constraint(160)) for ev in e]
                            
                            # Set tag to jazz
                            e = [ev.intersect({"tag": {"jazz", "modern"}}) for ev in e]
                            
                            # Add some optional drum events for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e