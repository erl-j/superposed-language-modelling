# write a drum and bass beat
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a drum and bass beat with a strong rhythm section.
                            '''
                            e = []
                            # Set up the drum pattern
                            # Kick drum (usually on beats 1 and 3)
                            for bar in range(4):
                                for beat in [0, 2]:
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/global_tick": {str(beat * 24 + bar * 96)}})
                                        .force_active()
                                    ]
                            
                            # Snare drum (usually on beats 2 and 4)
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/global_tick": {str(beat * 24 + bar * 96)}})
                                        .force_active()
                                    ]
                            
                            # Hi-hat pattern (on every 8th note)
                            for tick in range(0, 384, 12):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/global_tick": {str(tick)}})
                                    .force_active()
                                ]
                            
                            # Set up the bass line
                            # We'll create a simple, driving bass pattern
                            bass_pattern = [0, 7, 12, 7]  # Root, fifth, octave, fifth
                            for bar in range(4):
                                for beat, note in enumerate(bass_pattern):
                                    e += [
                                        ec()
                                        .intersect({
                                            "instrument": {"Bass"}, 
                                            "pitch": {str(36 + note)},  # E1 as root note
                                            "onset/global_tick": {str(beat * 24 + bar * 96)},
                                            "duration": {str(24)}  # Quarter note duration
                                        })
                                        .force_active()
                                    ]
                            
                            # Add some variation to the bass (optional notes)
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Bass"},
                                        "pitch": {str(pitch) for pitch in range(36, 60)}  # E1 to C3
                                    })
                                ]
                            
                            # Set tempo (assuming a typical DnB tempo)
                            e = [ev.intersect(ec().tempo_constraint(170)) for ev in e]
                            
                            # Set tag
                            e = [ev.intersect({"tag": {"dance-eletric"}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e