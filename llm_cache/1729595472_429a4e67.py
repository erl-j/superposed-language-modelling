# Create a jazzy drum beat
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            We want to create a jazzy drum beat with a focus on ride cymbal, snare, and kick drum.
                            We'll use syncopation and some ghost notes to give it a jazzy feel.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Add ride cymbal pattern (typically on every beat and some off-beats)
                            for beat in range(16):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"51 (Drums)"}, "onset/beat": {str(beat)}})
                                    .force_active()
                                ]
                                # Add some off-beat ride hits
                                if beat % 2 == 0:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"51 (Drums)"}, "onset/beat": {str(beat)}, "onset/tick": {"12"}})
                                        .force_active()
                                    ]
                            
                            # Add kick drum (on 1 and 3 of each bar, with some variations)
                            for bar in range(4):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4)}})
                                    .force_active()
                                ]
                                # Add some syncopated kicks
                                e += [
                                    ec()
                                    .intersect({"pitch": {"36 (Drums)"}, "onset/beat": {str(bar * 4 + 2)}, "onset/tick": {"12"}})
                                    .force_active()
                                ]
                            
                            # Add snare hits (on 2 and 4 of each bar)
                            for bar in range(4):
                                for beat in [1, 3]:
                                    e += [
                                        ec()
                                        .intersect({"pitch": {"38 (Drums)"}, "onset/beat": {str(bar * 4 + beat)}})
                                        .force_active()
                                    ]
                            
                            # Add some ghost notes on the snare
                            for _ in range(8):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"38 (Drums)"}, "instrument": {"Drums"}})
                                    .intersect(ec().velocity_constraint(30))
                                    .force_active()
                                ]
                            
                            # Add some hi-hat accents
                            for _ in range(6):
                                e += [
                                    ec()
                                    .intersect({"pitch": {"42 (Drums)"}, "instrument": {"Drums"}})
                                    .force_active()
                                ]
                            
                            # Set a jazzy tempo (around 120-140 BPM)
                            e = [ev.intersect(ec().tempo_constraint(130)) for ev in e]
                            
                            # Set tag to jazz
                            e = [ev.intersect({"tag": {"jazz"}}) for ev in e]
                            
                            # Add some optional drum hits for variety
                            e += [ec().intersect({"instrument": {"Drums"}}) for _ in range(10)]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e