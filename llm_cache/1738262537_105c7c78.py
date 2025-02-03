# make the drum beat more hip hop like
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a more hip-hop style drum beat. This typically involves:
                            - A strong, prominent kick drum pattern
                            - Snares on the 2 and 4
                            - A consistent hi-hat pattern
                            - Possibly some additional percussion elements
                            '''
                            # Remove inactive notes and isolate non-drum instruments
                            e = [ev for ev in e if ev.is_active()]
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Start over with the drum pattern
                            e = []
                            
                            # Add kick drum pattern (typically on 1 and 3, with some variations)
                            for bar in range(4):
                                for beat in [0, 2]:  # 1 and 3 of each bar
                                    e.append(ec().intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"},
                                                             "onset/global_tick": {str(beat * 24 + bar * 96)}}).force_active())
                                # Add an optional kick on the 4th beat for variation
                                e.append(ec().intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"},
                                                         "onset/global_tick": {str(3 * 24 + bar * 96)}}).force_active())
                            
                            # Add snares on 2 and 4
                            for bar in range(4):
                                for beat in [1, 3]:  # 2 and 4 of each bar
                                    e.append(ec().intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"},
                                                             "onset/global_tick": {str(beat * 24 + bar * 96)}}).force_active())
                            
                            # Add hi-hat pattern (eighth notes)
                            for tick in range(0, 384, 12):  # 12 ticks = eighth note
                                e.append(ec().intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"},
                                                         "onset/global_tick": {str(tick)}}).force_active())
                            
                            # Add some additional percussion elements (e.g., open hi-hat, tom, clap)
                            e += [ec().intersect({"instrument": {"Drums"}, "pitch": {"46 (Drums)", "50 (Drums)", "39 (Drums)"}}).force_active() for _ in range(8)]
                            
                            # Set velocity to create dynamics
                            e = [ev.intersect(ec().velocity_constraint(80)) for ev in e]  # General velocity
                            # Emphasize certain beats with higher velocity
                            for ev in e:
                                if ev.a["pitch"] == {"36 (Drums)"} or ev.a["pitch"] == {"38 (Drums)"}:
                                    ev.intersect(ec().velocity_constraint(100))
                            
                            # Preserve the original tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to hip-hop
                            e = [ev.intersect({"tag": {"hip-hop-rap"}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e