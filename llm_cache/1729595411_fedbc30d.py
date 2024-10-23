# Add a bassline with octave jumps
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to add a bassline with octave jumps to the existing beat.
                            '''
                            # Remove inactive notes and any existing bass notes
                            e = [ev for ev in e if ev.is_active() and ev.a["instrument"].isdisjoint({"Bass"})]
                            
                            # Define bass pitches for two octaves
                            low_octave = {str(pitch) for pitch in range(36, 48)}  # C1 to B1
                            high_octave = {str(pitch) for pitch in range(48, 60)}  # C2 to B2
                            
                            # Add bass notes on each beat, alternating between low and high octaves
                            for beat in range(16):  # 4 bars * 4 beats
                                octave = low_octave if beat % 2 == 0 else high_octave
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "onset/beat": {str(beat)},
                                        "pitch": octave
                                    })
                                    .force_active()
                                ]
                            
                            # Add some optional bass notes for variation
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": low_octave.union(high_octave)
                                    })
                                ]
                            
                            # Preserve the tempo from the existing beat
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the tag from the existing beat
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events to reach n_events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e