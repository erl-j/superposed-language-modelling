# A drill beat. One of the most distinctive features of drill drums is the hi-hat pattern, based on the tresillo rhythm. This rhythm, made up of two dotted eighth notes followed by an eighth note, creates a staggered, triplet feel.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a drill beat with the distinctive tresillo rhythm in the hi-hats.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set tempo to 140 BPM (typical for drill)
                            tempo_constraint = ec().tempo_constraint(140)
                            
                            # Kick drum pattern (typically on the 1 and in between 2 and 3)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}}).force_active(),
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}, "onset/tick": {"12"}}).force_active()
                                ]
                            
                            # Snare/Clap (typically on 2 and 4)
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)", "39 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
            
                            # Hi-hat tresillo pattern (two dotted eighth notes followed by an eighth note)
                            for bar in range(4):
                                base_beat = bar * 4
                                e += [
                                    ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(base_beat)}, "onset/tick": {"0"}}).force_active(),
                                    ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(base_beat + 1)}, "onset/tick": {"12"}}).force_active(),
                                    ec().intersect({"pitch": {"42 (Drums)"}, "onset/beat": {str(base_beat + 3)}, "onset/tick": {"0"}}).force_active(),
                                ]
            
                            # Add some open hi-hats for variation
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(4)]
            
                            # Add some 808-style bass notes (typical in drill)
                            e += [ec().intersect({"pitch": {"35 (Drums)"}}).force_active() for _ in range(8)]
            
                            # Add up to 20 optional drum notes for additional complexity
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
            
                            # Apply tempo constraint to all events
                            e = [ev.intersect(tempo_constraint) for ev in e]
            
                            # Set tag to 'electronic' (closest to drill in the given list)
                            e = [ev.intersect({"tag": {"electronic", "-"}}) for ev in e]
            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e