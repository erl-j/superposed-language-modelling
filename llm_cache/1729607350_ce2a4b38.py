# write a fast jungle drum beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a fast jungle drum beat with a focus on breakbeats, fast hi-hats, and syncopated rhythms.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set a fast tempo typical for jungle (around 160-180 BPM)
                            tempo = 170
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Add kick drum pattern (typically on the 1 and 3, with some variations)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar*4)}}).force_active(),
                                    ec().intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar*4 + 2)}}).force_active()
                                ]
                            # Add some syncopated kicks
                            e += [ec().intersect({"pitch": {"36 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add snare drum (typically on 2 and 4, with some variations)
                            for bar in range(4):
                                e += [
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar*4 + 1)}}).force_active(),
                                    ec().intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar*4 + 3)}}).force_active()
                                ]
                            # Add some syncopated snares
                            e += [ec().intersect({"pitch": {"38 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add fast hi-hats (closed)
                            e += [ec().intersect({"pitch": {"42 (Drums)"}}).force_active() for _ in range(32)]
                            
                            # Add some open hi-hats for variation
                            e += [ec().intersect({"pitch": {"46 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add some tom fills (typical in jungle breaks)
                            e += [ec().intersect({"pitch": {"45 (Drums)", "47 (Drums)", "48 (Drums)"}}).force_active() for _ in range(6)]
                            
                            # Add some cymbal crashes for accent
                            e += [ec().intersect({"pitch": {"49 (Drums)"}}).force_active() for _ in range(2)]
                            
                            # Add some optional percussion elements
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(20)]
                            
                            # Set velocity variations (some hits louder, some softer for dynamics)
                            e = [ev.intersect(ec().velocity_constraint(100)) for ev in e[:len(e)//2]]  # First half louder
                            e = [ev.intersect(ec().velocity_constraint(70)) for ev in e[len(e)//2:]]   # Second half softer
                            
                            # Set tag to appropriate genre
                            e = [ev.intersect({"tag": {"dance-eletric", "-"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e