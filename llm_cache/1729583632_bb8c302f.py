# add some piano chords
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add piano chords to the existing beat. We'll add chords on the first beat of each bar,
                            and some additional optional piano notes for variation.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove any existing piano notes
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
                                    for _ in range(3)  # Add 3 notes to form a chord
                                ]
                            
                            # Add up to 12 optional piano notes for variation
                            e += [
                                ec()
                                .intersect({
                                    "instrument": {"Piano"},
                                    "pitch": {str(pitch) for pitch in range(60, 85)},  # Middle to high range for piano
                                })
                                for _ in range(12)
                            ]
                            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e