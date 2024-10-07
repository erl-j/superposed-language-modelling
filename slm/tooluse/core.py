from paper_checkpoints import checkpoints
from train import EncoderOnlyModel
from util import get_scale

device = "cuda:6"
ROOT_DIR = "./"
MODEL = "slm"
if MODEL == "slm":
    model = (
        EncoderOnlyModel.load_from_checkpoint(
            ROOT_DIR + checkpoints[MODEL],
            map_location=device,
        )
        .to(device)
        .eval()
    )

    def generate(
        mask,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        tokens_per_step=1,
        attribute_temperature=None,
        order=None,
    ):
        return model.generate(
            mask,
            temperature=temperature,
            tokens_per_step=tokens_per_step,
            top_p=top_p,
            top_k=top_k,
            order=order,
            # attribute_temperature={"velocity": 1.5,"onset/tick":0.5},
        )[0].argmax(axis=1)

ALL_INSTRUMENTS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "instrument:" in token and token != "instrument:-"
}

ALL_TEMPOS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "tempo:" in token and token != "tempo:-"
}
ALL_ONSET_BEATS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "onset/beat:" in token and token != "onset/beat:-"
}
ALL_ONSET_TICKS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "onset/tick:" in token and token != "onset/tick:-"
}
ALL_OFFSET_BEATS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "offset/beat:" in token and token != "offset/beat:-"
}
ALL_OFFSET_TICKS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "offset/tick:" in token and token != "offset/tick:-"
}
ALL_VELOCITIES = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "velocity:" in token and token != "velocity:-"
}
ALL_TAGS = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "tag:" in token and token != "tag:-"
}
ALL_PITCH = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "pitch:" in token and token != "pitch:-"
}
# all pitches that contain "(Drums)"
DRUM_PITCHES = {
    token.split(":")[-1]
    for token in model.tokenizer.vocab
    if "pitch:" in token and "(Drums)" in token
}

HIHAT_PITCHES = {f"{pitch} (Drums)" for pitch in ["42", "44", "46"]}

TOM_PITCHES = {f"{pitch} (Drums)" for pitch in ["48", "50", "45", "47"]}

CRASH_PITCHES = {f"{pitch} (Drums)" for pitch in ["49", "57"]}

PERCUSSION_PITCHES = {
    f"{pitch} (Drums)"
    for pitch in [
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
    ]
}

# create 128 bpm rock loop with drums, bass, guitar with max 280 notes
N_EVENTS = model.tokenizer.config["max_notes"]

def sm_to_events(x_sm):
    x = model.tokenizer.encode(x_sm, tag="other")
    tokens = model.tokenizer.indices_to_tokens(x)
    # group by n_attributes
    n_attributes = len(model.tokenizer.note_attribute_order)
    # n_events = model.tokenizer.config["max_notes"]
    events = []
    for i in range(0, len(tokens), n_attributes):
        event = {key: set() for key in model.tokenizer.note_attribute_order}
        for j in range(n_attributes):
            token = tokens[i + j]
            key, value = token.split(":")
            event[key].add(value)
        events.append(event)
    # create event objects
    events = [EventConstraint().intersect(event) for event in events]
    return events

class EventConstraint:
    def __init__(self, dict=None):
        self.blank_event = {
            "instrument": ALL_INSTRUMENTS | {"-"},
            "pitch": ALL_PITCH | {"-"},
            "onset/beat": ALL_ONSET_BEATS | {"-"},
            "onset/tick": ALL_ONSET_TICKS | {"-"},
            "offset/beat": ALL_OFFSET_BEATS | {"-"},
            "offset/tick": ALL_OFFSET_TICKS | {"-"},
            "velocity": ALL_VELOCITIES | {"-"},
            "tag": ALL_TAGS | {"-"},
            "tempo": ALL_TEMPOS | {"-"},
        }

        self.a = self.blank_event.copy()

        self.all_active = {
            "instrument": ALL_INSTRUMENTS,
            "pitch": ALL_PITCH,
            "onset/beat": ALL_ONSET_BEATS,
            "onset/tick": ALL_ONSET_TICKS,
            "offset/beat": ALL_OFFSET_BEATS,
            "offset/tick": ALL_OFFSET_TICKS,
            "velocity": ALL_VELOCITIES,
            "tag": ALL_TAGS,
            "tempo": ALL_TEMPOS,
        }

        self.not_active = {
            "instrument": {"-"},
            "pitch": {"-"},
            "onset/beat": {"-"},
            "onset/tick": {"-"},
            "offset/beat": {"-"},
            "offset/tick": {"-"},
            "velocity": {"-"},
            "tag": {"-"},
            "tempo": {"-"},
        }

        if dict is not None:
            self.a = dict

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


def scale_constraint(scale, pitch_range):
    scale_constraint = {
        "pitch": {str(note) for note in get_scale(scale, pitch_range)} | {"-"}
    }
    return scale_constraint


def tempo_constraint(tempo):
    # find tempo that is closest to the given tempo
    tempos = list(str(t) for t in ALL_TEMPOS)
    tempo = min(tempos, key=lambda x: abs(int(x) - tempo))
    return {"tempo": {tempo, "-"}}


def velocity_constraint(velocity):
    velocities = list(str(v) for v in ALL_VELOCITIES)
    velocity = min(velocities, key=lambda x: abs(int(x) - velocity))
    return {"velocity": {velocity, "-"}}


def quantize_velocity(velocity):
    velocities = list(str(v) for v in ALL_VELOCITIES)
    velocity = min(velocities, key=lambda x: abs(int(x) - int(velocity)))
    return velocity

def quantize_tempo(tempo):
    tempos = list(str(t) for t in ALL_TEMPOS)
    tempo = min(tempos, key=lambda x: abs(int(x) - int(tempo)))
    return tempo
