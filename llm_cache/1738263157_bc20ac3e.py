# add some syncopated tom dominant drum beat
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a syncopated tom-dominant drum beat. We'll use various tom pitches,
                            add some kicks and hi-hats for rhythm, and ensure syncopation by placing some hits on off-beats.
                            '''
                            e = [ev for ev in e if ev.is_active()]
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            e = []

                            # Define tom pitches (low, mid, high)
                            tom_pitches = {"41 (Drums)", "45 (Drums)", "48 (Drums)"}

                            # Add syncopated tom hits
                            for bar in range(4):
                                for beat in range(4):
                                    # On-beat toms
                                    e.append(ec().intersect({
                                        "instrument": {"Drums"},
                                        "pitch": tom_pitches,
                                        "onset/global_tick": {str(bar * 96 + beat * 24)}
                                    }).force_active())
                                    
                                    # Off-beat toms (on the "and" of each beat)
                                    if beat != 3:  # Skip the last off-beat of each bar
                                        e.append(ec().intersect({
                                            "instrument": {"Drums"},
                                            "pitch": tom_pitches,
                                            "onset/global_tick": {str(bar * 96 + beat * 24 + 12)}
                                        }).force_active())

            
                            # Add kicks on 1 and 3 of each bar
                            for bar in range(4):
                                for beat in [0, 2]:
                                    e.append(ec().intersect({
                                        "instrument": {"Drums"},
                                        "pitch": {"36 (Drums)"},
                                        "onset/global_tick": {str(bar * 96 + beat * 24)}
                                    }).force_active())

                            # Add hi-hats for rhythm
                            for i in range(16):  # 16 eighth notes in 4 bars
                                e.append(ec().intersect({
                                    "instrument": {"Drums"},
                                    "pitch": {"42 (Drums)"},
                                    "onset/global_tick": {str(i * 12)}
                                }).force_active())

                            # Add some optional drum hits for variation
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]

                            # Set tempo to current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]

                            # Set tag to current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]

                            # Add back other instruments
                            e += e_other

                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]

                            return e