from util import get_scale

class EventConstraint:
    '''
    This class is used to represent constraints on events.
    The constraints are represented as a dictionary with keys as event attributes
    and values as sets of possible values for the attribute.
    The class provides methods to intersect and union constraints.
    The class also provides methods to check if the event is active or inactive.
    The class also provides methods to force the event to be active or inactive.
    '''
    def __init__(self, blank_event):
        self.blank_event = blank_event.copy()
        # everything except "-"
        self.all_active = {
            key: set(values) - {"-"} for key, values in self.blank_event.items()
        }
        self.not_active = {key: {"-"} for key in self.blank_event}
        self.a = self.blank_event.copy()

    def intersect(self, constraint):
        for key in constraint:
            self.a[key] = self.a[key] & constraint[key]
        for key in self.a:
            assert (
                len(self.a[key]) > 0
            ), f"Empty set for key {key}, constraint {constraint}, attributes {self.a}"
        return self

    def union(self, constraint):
        for key in constraint:
            self.a[key] = self.a[key] | constraint[key]
        return self

    def is_inactive(self):
        for key in self.a:
            if self.a[key] != {"-"}:
                return False
        return True

    def is_active(self):
        for key in self.a:
            if self.a[key] == {"-"}:
                return False
        return True

    def force_active(self):
        self.intersect(self.all_active)
        return self

    def force_inactive(self):
        self.intersect(self.not_active)
        return self

    def to_dict(self):
        return self.a
    
class MusicalEventConstraint(EventConstraint):
    '''
    This class extends the EventConstraint class to provide some utilities for musical events.
    '''

    def __init__(self, blank_event):
        super().__init__(blank_event)

    def pitch_in_scale_constraint(self,scale, pitch_range):
        scale_constraint = {
            "pitch": {str(note) for note in get_scale(scale, pitch_range)} | {"-"}
        }
        return scale_constraint

    def tempo_constraint(self,tempo):
        # find tempo that is closest to the given tempo
        tempos = list(str(t) for t in self.blank_event["tempo"] if t != "-")
        tempo = min(tempos, key=lambda x: abs(int(x) - tempo))
        return {"tempo": {tempo, "-"}}

    def velocity_constraint(self,velocity):
        velocities = list(str(v) for v in self.blank_event["velocity"] if v != "-")
        velocity = min(velocities, key=lambda x: abs(int(x) - velocity))
        return {"velocity": {velocity, "-"}}

    def quantize_velocity(self,velocity):
        velocities = list(str(v) for v in self.blank_event["velocity"] if v != "-")
        velocity = min(velocities, key=lambda x: abs(int(x) - int(velocity)))
        return velocity

    def quantize_tempo(self,tempo):
        tempos = list(str(t) for t in self.blank_event["tempo"] if t != "-")
        tempo = min(tempos, key=lambda x: abs(int(x) - int(tempo)))
        return tempo

DRUM_PITCHES = {
    f"{pitch} (Drums)"
    for pitch in range(35, 82)
}

HIHAT_PITCHES = {f"{pitch} (Drums)" for pitch in ["42", "44", "46"]}

TOM_PITCHES = {f"{pitch} (Drums)" for pitch in ["48", "50", "45", "47"]}

CRASH_PITCHES = {f"{pitch} (Drums)" for pitch in ["49", "57"]}

PERCUSSION_PITCHES = {
    f"{pitch} (Drums)"
    for pitch in range(54, 82)
}