# create a drum and bass beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a drum and bass beat with a fast-paced drum pattern and a syncopated bassline.
                            '''
                            e = []
                            # Set up the drum pattern
                            # Kick drum
                            for beat in range(16):
                                if beat % 4 == 0:  # on every quarter note
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(beat)}})
                                        .force_active()
                                    ]
                            
                            # Snare drum
                            for beat in [2, 6, 10, 14]:  # on 2 and 4 of each bar
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Hi-hat pattern
                            for beat in range(16):
                                for tick in [0, 12]:  # sixteenth notes
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {str(tick)}})
                                        .force_active()
                                    ]
                            
                            # Bassline
                            bass_pattern = [0, 4, 7, 11, 14]  # syncopated rhythm
                            for beat in bass_pattern:
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Bass"}, "onset/beat": {str(beat)}, "pitch": {str(p) for p in range(30, 50)}})
                                    .force_active()
                                ]
                            
                            # Add some optional bass notes
                            e += [ec().intersect({"instrument": {"Bass"}, "pitch": {str(p) for p in range(30, 50)}}) for _ in range(5)]
                            
                            # Set tempo to a typical drum and bass tempo
                            e = [ev.intersect(ec().tempo_constraint(170)) for ev in e]
                            
                            # Set tag to dance-eletric (closest to drum and bass)
                            e = [ev.intersect({"tag": {"dance-eletric"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e