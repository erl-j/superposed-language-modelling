# Add some piano chords
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add piano chords to the existing beat. We'll place chords on the downbeats of each bar
                            and add some optional notes for variation.
                            '''
                            # Remove inactive notes
                            e = [ev for ev in e if ev.is_active()]
                            
                            # Remove any existing piano notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Add piano chords on the downbeat of each bar
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
            
                            # Add some optional piano notes for variation
                            e += [
                                ec()
                                .intersect({"instrument": {"Piano"}})
                                for _ in range(10)
                            ]
            
                            # Ensure the piano notes are in a reasonable pitch range
                            e = [ev.intersect({"pitch": {str(p) for p in range(60, 85)}}) for ev in e]
            
                            # Preserve the current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
            
                            # Preserve the current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
            
                            return e