# add some piano chords
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add some piano chords to the existing funk beat and bassline.
                            The piano chords will be added on the first beat of each bar, with some additional optional chords.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            # Remove existing piano notes if any
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Add piano chords on the first beat of each bar
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(bar * 4)},
                                        "offset/beat": {str(bar * 4 + 2)},  # Make the chord last for 2 beats
                                    })
                                    .force_active()
                                    for _ in range(3)  # Add 3 notes for each chord
                                ]
                            
                            # Add some optional piano chords
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(beat) for beat in range(16)},
                                    })
                                ]
                            
                            # Set pitch range for piano
                            e = [ev.intersect({"pitch": {str(pitch) for pitch in range(48, 84)}}) if ev.a["instrument"] == {"Piano"} else ev for ev in e]
                            
                            # Set velocity for piano (slightly lower than the drums for balance)
                            e = [ev.intersect(ec().velocity_constraint(80)) if ev.a["instrument"] == {"Piano"} else ev for ev in e]
                            
                            # Preserve tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e