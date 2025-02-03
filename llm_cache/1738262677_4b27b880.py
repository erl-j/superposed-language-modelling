# make the drum beat more hip hop like
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a more hip-hop like drum beat. This typically involves:
                            - A strong, prominent kick drum pattern
                            - Snares on the 2 and 4
                            - A consistent hi-hat pattern
                            - Possibly some additional percussion elements
                            '''
                            # Remove inactive notes and isolate non-drum instruments
                            e = [ev for ev in e if ev.is_active()]
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Start with a clean slate for drums
                            e = []
                            
                            # Add kick drum pattern (typically on 1 and 3, with some variations)
                            for bar in range(4):
                                for beat in [0, 2]:  # 1 and 3 of each bar
                                    e.append(ec().intersect({"pitch": {"36 (Drums)"}, 
                                                             "onset/global_tick": {str(beat * 24 + bar * 96)}})
                                             .force_active())
                                # Add an optional kick on the 4th beat for variation
                                e.append(ec().intersect({"pitch": {"36 (Drums)"}, 
                                                         "onset/global_tick": {str(3 * 24 + bar * 96)}})
                                         .force_active())
                            
                            # Add snares on 2 and 4
                            for bar in range(4):
                                for beat in [1, 3]:  # 2 and 4 of each bar
                                    e.append(ec().intersect({"pitch": {"38 (Drums)"}, 
                                                             "onset/global_tick": {str(beat * 24 + bar * 96)}})
                                             .force_active())
                            
                            # Add a consistent hi-hat pattern (eighth notes)
                            for tick in range(0, 384, 12):  # 12 ticks = eighth note
                                e.append(ec().intersect({"pitch": {"42 (Drums)"}, 
                                                         "onset/global_tick": {str(tick)}})
                                         .force_active())
                            
                            # Add some optional percussion elements
                            e += [ec().intersect({"pitch": {"37 (Drums)", "39 (Drums)", "43 (Drums)", "45 (Drums)"}}) 
                                  for _ in range(8)]
                            
                            # Set velocity for a more dynamic feel
                            e = [ev.intersect(ec().velocity_constraint(80)) for ev in e]
                            
                            # Preserve the original tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to hip-hop
                            e = [ev.intersect({"tag": {"hip-hop-rap", "-"}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e