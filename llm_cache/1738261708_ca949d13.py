# add a ride
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a ride cymbal pattern to the existing beat.
                            The ride cymbal will provide a steady rhythm, typically on quarter notes or eighth notes.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Start with existing drum events
                            e_drums = [ev for ev in e if not ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add ride cymbal pattern
                            # Ride cymbal is typically MIDI note 51 or 53
                            ride_pitches = {"51 (Drums)", "53 (Drums)"}
                            
                            # Add ride on every quarter note (every 24 ticks)
                            for tick in range(0, 384, 24):
                                e_drums.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Drums"},
                                        "pitch": ride_pitches,
                                        "onset/global_tick": {str(tick)}
                                    })
                                    .force_active()
                                )
                            
                            # Add some variations (optional eighth notes)
                            for _ in range(8):
                                e_drums.append(
                                    ec()
                                    .intersect({
                                        "instrument": {"Drums"},
                                        "pitch": ride_pitches
                                    })
                                )
                            
                            # Combine all events
                            e = e_drums + e_other
                            
                            # Preserve tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e