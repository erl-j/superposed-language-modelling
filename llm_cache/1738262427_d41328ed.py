# make the drum beat more sparse and unhinged
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a more sparse and unhinged drum beat. We'll reduce the number of drum hits,
                            add some off-beat elements, and introduce more variety in the drum sounds used.
                            '''
                            # Remove inactive notes and isolate drums
                            e = [ev for ev in e if ev.is_active()]
                            e_drums = [ev for ev in e if ev.a["instrument"].intersection({"Drums"})]
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Start over with drums
                            e_drums = []
                            
                            # Add sparse kick drum (4-6 hits)
                            e_drums += [ec().intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}}).force_active() for _ in range(random.randint(4, 6))]
                            
                            # Add sparse snare (3-5 hits, some potentially off-beat)
                            for _ in range(random.randint(3, 5)):
                                e_drums.append(
                                    ec().intersect({"instrument": {"Drums"}, "pitch": {"38 (Drums)"}})
                                    .intersect({"onset/global_tick": {str(tick) for tick in range(0, 384, 12)}})  # Allow for off-beat placements
                                    .force_active()
                                )
                            
                            # Add occasional tom hits (2-4 hits)
                            tom_pitches = {"45 (Drums)", "47 (Drums)", "50 (Drums)"}  # Low, mid, and high toms
                            e_drums += [
                                ec().intersect({"instrument": {"Drums"}, "pitch": tom_pitches})
                                .intersect({"onset/global_tick": {str(tick) for tick in range(0, 384, 6)}})  # Even more off-beat possibilities
                                .force_active() 
                                for _ in range(random.randint(2, 4))
                            ]
                            
                            # Add sparse hi-hat (open and closed, 5-8 hits)
                            hihat_pitches = {"42 (Drums)", "46 (Drums)"}  # Closed and open hi-hat
                            e_drums += [
                                ec().intersect({"instrument": {"Drums"}, "pitch": hihat_pitches})
                                .intersect({"onset/global_tick": {str(tick) for tick in range(0, 384, 8)}})  # Some syncopation
                                .force_active() 
                                for _ in range(random.randint(5, 8))
                            ]
                            
                            # Add a couple of crash cymbals for dramatic effect
                            e_drums += [
                                ec().intersect({"instrument": {"Drums"}, "pitch": {"49 (Drums)"}})  # Crash cymbal
                                .force_active() 
                                for _ in range(2)
                            ]
                            
                            # Combine drums with other instruments
                            e = e_drums + e_other
                            
                            # Preserve tempo and tag
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e