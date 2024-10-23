# rewrite the drums
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to rewrite the drums while keeping the existing tempo and tag.
                            We'll create a new drum pattern with kicks, snares, hi-hats, and some additional percussion.
                            '''
                            # Remove inactive notes and isolate non-drum instruments
                            e = [ev for ev in e if ev.is_active()]
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Start with an empty list for our new drum pattern
                            e = []
                            
                            # Add kicks on beats 1 and 3 of each bar
                            for bar in range(4):
                                for beat in [0, 2]:
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add snares on beats 2 and 4 of each bar
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}, "onset/beat": {str(beat + bar * 4)}})
                                        .force_active()
                                    ]
                            
                            # Add hi-hats on every beat
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Add some off-beat hi-hats
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
                            
                            # Add some tom fills in the last bar
                            for beat in [12, 13, 14, 15]:
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"45 (Drums)", "47 (Drums)", "50 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                            
                            # Add up to 10 optional percussion hits
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Preserve the original tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Preserve the original tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with inactive notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e