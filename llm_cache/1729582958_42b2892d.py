# Replace the piano with some short piano chords
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to replace the existing piano with short piano chords.
                            '''
                            # Remove existing piano notes
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Piano"})]
                            
                            # Add short piano chords
                            for beat in range(16):  # 16 beats in a 4-bar loop
                                e += [
                                    ec()
                                    .intersect({
                                        "instrument": {"Piano"},
                                        "onset/beat": {str(beat)},
                                        "offset/beat": {str(beat)},  # Same as onset for short chord
                                        "offset/tick": {"12"},  # Half a beat duration
                                    })
                                    .force_active()
                                    for _ in range(3)  # 3 notes per chord
                                ]
                            
                            # Set velocity for piano chords (slightly accented on beats 1 and 3 of each bar)
                            for ev in e:
                                if "Piano" in ev.a["instrument"]:
                                    if ev.a["onset/beat"].issubset({"0", "4", "8", "12"}):
                                        ev.intersect(ec().velocity_constraint(90))
                                    else:
                                        ev.intersect(ec().velocity_constraint(70))
                            
                            # Preserve tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add some optional piano notes for variation
                            e += [ec().intersect({"instrument": {"Piano"}}) for _ in range(10)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e