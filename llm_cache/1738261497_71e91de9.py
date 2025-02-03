# add some rides
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add ride cymbals to the existing beat. Rides will be added to complement
                            the existing rhythm, typically on quarter notes or eighth notes.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Find existing drum events
                            e_drums = [ev for ev in e if not ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add ride cymbals
                            ride_events = []
                            
                            # Add a ride on each quarter note (every 96 ticks)
                            for i in range(0, 384, 96):
                                ride_events.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Drums"},
                                        "pitch": {"51 (Drums)"}, # 51 is typically used for ride cymbal
                                        "onset/global_tick": {str(i)}
                                    })
                                    .force_active()
                                )
                            
                            # Add some additional rides on eighth notes (every 48 ticks)
                            for i in range(48, 384, 96):
                                ride_events.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Drums"},
                                        "pitch": {"51 (Drums)"},
                                        "onset/global_tick": {str(i)}
                                    })
                                )
                            
                            # Combine all events
                            e = e_drums + ride_events + e_other
                            
                            # Set tempo to current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e