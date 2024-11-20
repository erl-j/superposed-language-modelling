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
    
    def __str__(self):
        return str(self.a)
    
class MusicalEventConstraint(EventConstraint):
    '''
    This class extends the EventConstraint class to provide some utilities for musical events.
    '''

    def __init__(self, blank_event, tokenizer):
        super().__init__(blank_event)
        self.tokenizer = tokenizer

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

    def sanitize_time(self):
        """
        Sanitizes the time constraints according to these rules:
        1. If onset is determined (single numerical value), filter offsets to be greater than onset or "none (Drums)"
        2. If offset is determined (single numerical value), filter onsets to be less than offset
        """
        # Check if onset is determined and numerical
        if "onset/global_tick" in self.a and len(self.a["onset/global_tick"]) == 1:
            onset_value = next(iter(self.a["onset/global_tick"]))
            if onset_value.isdigit():  # Check if it's a numerical value
                onset_num = int(onset_value)
                if "offset/global_tick" in self.a:
                    print(f"current onset: {onset_num}")
                    # print old offsets
                    print("Old offsets: ", sorted(list(self.a["offset/global_tick"])))
                    # Keep only valid offsets (greater than onset) or "none (Drums)"
                    valid_offsets = {
                        offset for offset in self.a["offset/global_tick"]
                        if offset == "none (Drums)" or (offset.isdigit() and int(offset) > onset_num)
                    }
                    self.a["offset/global_tick"] = valid_offsets
                    assert len(valid_offsets) > 0, f"No valid offsets found for onset {onset_num}"
                    # print new offsets
                    print("New offsets: ", sorted(list(self.a["offset/global_tick"])))

        # Check if offset is determined and numerical
        elif "offset/global_tick" in self.a and len(self.a["offset/global_tick"]) == 1:
            offset_value = next(iter(self.a["offset/global_tick"]))
            if offset_value.isdigit():  # Check if it's a numerical value
                offset_num = int(offset_value)
                if "onset/global_tick" in self.a:
                    # Keep only valid onsets (less than offset)
                    valid_onsets = {
                        onset for onset in self.a["onset/global_tick"]
                        if onset.isdigit() and int(onset) < offset_num
                    }
                    self.a["onset/global_tick"] = valid_onsets
                    assert len(valid_onsets) > 0, f"No valid onsets found for offset {offset_num}"

        return self
    
    def sanitize_undef(self):
        """
        Sanitizes the constraint according to undefined state rules:
        1. If any attribute is undefined {"-"}, all attributes become undefined
        2. If any attribute is defined (not {"-"}), remove "-" from all other attributes
        """
        has_undefined = False
        has_defined = False
        
        # Check if we have any undefined or defined attributes
        for key in self.a:
            if self.a[key] == {"-"}:
                has_undefined = True
            elif "-" not in self.a[key]:
                has_defined = True
                
        # Rule 1: If any attribute is undefined, make all undefined
        if has_undefined:
            for key in self.a:
                self.a[key] = {"-"}
                
        # Rule 2: If any attribute is fully defined, remove "-" from all
        elif has_defined:
            for key in self.a:
                self.a[key] = self.a[key] - {"-"}
                
        # Verify we haven't created any empty sets
        for key in self.a:
            assert len(self.a[key]) > 0, f"Empty set for key {key} after sanitization"
                
        return self


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