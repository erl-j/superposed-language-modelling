# jungle drum beat
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a jungle drum beat, which typically features fast breakbeats, 
                            syncopated rhythms, and prominent use of bass drums and snares.
                            '''
                            e = []
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add a fast breakbeat pattern
                            # Kick drum (usually on the 1 and 3)
                            for bar in range(4):
                                for beat in [0, 2]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"36 (Drums)"}, "onset/global_tick": {str(beat * 24 + bar * 96)}})
                                        .force_active()
                                    ]
                            
                            # Snare drum (usually on the 2 and 4, with some syncopation)
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/global_tick": {str(beat * 24 + bar * 96)}})
                                        .force_active()
                                    ]
                            # Add some syncopated snares
                            e += [
                                ec()
                                .intersect({"pitch": {"38 (Drums)"}, "onset/global_tick": {str(12 + bar * 96)}})
                                .force_active()
                                for bar in range(4)
                            ]
                            
                            # Hi-hats (16th note pattern)
                            for tick in range(0, 384, 6):  # 16th notes
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "onset/global_tick": {str(tick)}})
                                    .force_active()
                                ]
                            
                            # Add some tom fills
                            e += [ec().intersect({"pitch": {"45 (Drums)", "47 (Drums)", "50 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add some crash cymbals for accents
                            e += [
                                ec()
                                .intersect({"pitch": {"49 (Drums)"}, "onset/global_tick": {"0", "192"}})
                                .force_active()
                                for _ in range(2)
                            ]
                            
                            # Set a fast tempo typical for jungle (160-180 BPM)
                            jungle_tempo = 170
                            e = [ev.intersect(ec().tempo_constraint(jungle_tempo)) for ev in e]
                            
                            # Set tag to electronic/dance
                            e = [ev.intersect({"tag": {"dance-eletric"}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e