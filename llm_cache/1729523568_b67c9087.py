# Now add some guitar chords
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add guitar chords to the existing funk beat and bassline.
                            The guitar chords will be added on the downbeats of each bar, with some variations.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            # Remove any existing guitar notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Guitar"})]
                            
                            # Add guitar chords on the downbeats (first beat of each bar)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Guitar"},
                                        "onset/beat": {str(bar * 4)},
                                        "offset/beat": {str(bar * 4 + 2)},  # Make the chord last for 2 beats
                                        "pitch": {str(pitch) for pitch in range(50, 70)}  # Mid-range pitches for guitar chords
                                    })
                                    .force_active()
                                ]
                            
                            # Add some variations - shorter chords on beats 2 and 4 of bars 2 and 4
                            for bar in [1, 3]:
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Guitar"},
                                            "onset/beat": {str(bar * 4 + beat)},
                                            "offset/beat": {str(bar * 4 + beat + 1)},  # Make these chords last for 1 beat
                                            "pitch": {str(pitch) for pitch in range(50, 70)}
                                        })
                                        .force_active()
                                    ]
                            
                            # Add up to 5 optional guitar notes for additional variation
                            e += [ec().intersect({"instrument": {"Guitar"}}) for _ in range(5)]
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag (assuming it's still "funk")
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e