# add some syncopated tom dominant drum beat
def build_constraint(e, ec, n_events, tick_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a syncopated tom-dominant drum beat. We'll use various toms, 
                            add some kicks and hi-hats for rhythm, and create syncopation by placing some hits 
                            on off-beats.
                            '''
                            e = []
                            # Set aside other instruments
                            e_other = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add toms (low, mid, high)
                            tom_pitches = {"41 (Drums)", "45 (Drums)", "48 (Drums)"}
                            for _ in range(20):  # Add 20 tom hits
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": tom_pitches})
                                    .force_active()
                                ]
                            
                            # Add some kicks for foundation (less than usual for tom-dominant beat)
                            for _ in range(6):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"36 (Drums)"}})
                                    .force_active()
                                ]
                            
                            # Add some hi-hats for rhythm
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "pitch": {"42 (Drums)"}})
                                    .force_active()
                                ]
                            
                            # Create syncopation by placing some hits on off-beats
                            syncopated_ticks = {6, 18, 30, 42, 54, 66, 78, 90}  # Off-beat ticks
                            for tick in syncopated_ticks:
                                e += [
                                    ec()
                                    .intersect({"instrument": {"Drums"}, "onset/global_tick": {str(tick)}})
                                    .force_active()
                                ]
                            
                            # Add some optional drum hits for variety
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Set tempo to current tempo
                            e = [ev.intersect(ec().tempo_constraint(tempo)) for ev in e]
                            
                            # Set tag to current tag
                            e = [ev.intersect({"tag": {tag}}) for ev in e]
                            
                            # Add back other instruments
                            e += e_other
                            
                            # Pad with empty notes
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e