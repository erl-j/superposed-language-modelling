# A drill beat. One of the most distinctive features of drill drums is the hi-hat pattern, based on the tresillo rhythm. This rhythm, made up of two dotted eighth notes followed by an eighth note, creates a staggered, triplet feel.
def build_constraint(e, ec, n_events, beat_range, pitch_range, drums, tag, tempo):
                            '''
                            To generate a drill beat with the distinctive tresillo rhythm in the hi-hats.
                            '''
                            e = []
                            # Remove all existing drums
                            e = [ev for ev in e if ev.a["instrument"].isdisjoint({"Drums"})]
                            
                            # Set tempo to 140 BPM (typical for drill)
                            tempo_constraint = ec().tempo_constraint(140)
                            
                            # Kick drum pattern (typically on the 1 and in between 2 and 3)
                            kick_pattern = [0, 10] # in 16th notes
                            for beat in range(4):
                                for tick in kick_pattern:
                                    e.append(ec().intersect({"pitch": {"36 (Drums)"}, 
                                                             "onset/beat": {str(beat)},
                                                             "onset/tick": {str(tick * 6)}}).force_active())
                            
                            # Snare/Clap (typically on beat 3)
                            for bar in range(4):
                                e.append(ec().intersect({"pitch": {"38 (Drums)", "39 (Drums)"},  # Both snare and clap sounds
                                                         "onset/beat": {str(2 + bar * 4)}}).force_active())
                            
                            # Hi-hat tresillo pattern (3+3+2 rhythm)
                            hi_hat_pattern = [0, 9, 18] # in 32nd notes
                            for bar in range(4):
                                for beat in range(4):
                                    for tick in hi_hat_pattern:
                                        e.append(ec().intersect({"pitch": {"42 (Drums)", "46 (Drums)"},  # Both closed and open hi-hats
                                                                 "onset/beat": {str(beat + bar * 4)},
                                                                 "onset/tick": {str(tick * 3)}}).force_active())
                            
                            # Add some variation with occasional 808 bass hits
                            for _ in range(8):
                                e.append(ec().intersect({"pitch": {"35 (Drums)"}}).force_active())
                            
                            # Add some optional percussion hits
                            for _ in range(10):
                                e.append(ec().intersect({"pitch": {"37 (Drums)", "40 (Drums)", "41 (Drums)"}})  # Side stick, electric snare, low floor tom
                                         .intersect(ec().velocity_constraint(60)))  # Lower velocity for subtle effect
                            
                            # Set tag to 'electronic' and 'urban' as drill is a subgenre of trap which falls under these categories
                            tag_constraint = ec().intersect({"tag": {"electronic", "urban"}})
                            
                            # Apply tempo and tag constraints to all events
                            e = [ev.intersect(tempo_constraint).intersect(tag_constraint) for ev in e]
                            
                            # Pad with inactive events
                            e += [ec().force_inactive() for _ in range(n_events - len(e))]
                            
                            return e