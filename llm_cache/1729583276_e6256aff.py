# add some dancy 909 drums
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add some dancy 909-style drums to the existing beat.
                            The Roland TR-909 is known for its punchy kick, crisp snare, and distinctive hi-hats.
                            We'll add these elements while preserving the existing beat structure.
                            '''
                            # Preserve existing events
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Add 909-style kick drum (36 is typically used for 909 kick)
                            e += [ec().intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Add 909-style snare (40 is often used for 909 snare)
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {"40 (Drums)"}, "onset/beat": {str(beat)} })
                                .force_active()
                                for beat in [2, 6, 10, 14]  # Typical snare on 2 and 4 of each bar
                            ]
                            
                            # Add closed hi-hats (42 for closed hi-hat)
                            e += [ec().intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}}).force_active() for _ in range(16)]
                            
                            # Add open hi-hats (46 for open hi-hat)
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {"46 (Drums)"}, "onset/beat": {str(beat)} })
                                .force_active()
                                for beat in [3, 7, 11, 15]  # Typical placement for open hi-hats
                            ]
                            
                            # Add some claps (39 is often used for clap sound)
                            e += [ec().intersect({"instrument": {"Drums"}, "pitch": {"39 (Drums)"}}).force_active() for _ in range(4)]
                            
                            # Add some tom fills (45, 47, 50 for low, mid, high toms)
                            tom_pitches = ["45 (Drums)", "47 (Drums)", "50 (Drums)"]
                            e += [
                                ec()
                                .intersect({"instrument": {"Drums"}, "pitch": {tom_pitch}})
                                .force_active()
                                for tom_pitch in tom_pitches
                            ]
                            
                            # Add some optional percussion hits
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Preserve the original tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Update the tag to reflect the dance style
                            e = [ev.intersect({"tag": {"dance-eletric", tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e