import numpy as np
def piano_roll(sm):
    sm = sm.copy()
    sm = sm.resample(tpq=4, min_dur=0)

    # set all is_drum to False
    for track in sm.tracks:
        track.is_drum = False

    pr = sm.pianoroll(modes=["frame"]).sum(axis=0).sum(axis=0)


    return pr



