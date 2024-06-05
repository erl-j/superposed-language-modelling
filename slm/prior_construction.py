#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data import MidiDataset
from train import EncoderOnlyModel
from util import preview_sm, sm_fix_overlap_notes, get_scale, loop_sm
import os
import IPython.display as ipd
from paper_checkpoints import checkpoints
from simplex_diffusion import SimplexDiffusionModel
import torch

device = "cuda:0"
ROOT_DIR = "../"

MODEL = "slm_drum"

OUTPUT_DIR = ROOT_DIR + "artefacts/examples_4"
TMP_DIR = ROOT_DIR + "artefacts/tmp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if MODEL == "slm":
    model = (
        EncoderOnlyModel.load_from_checkpoint(
            ROOT_DIR + checkpoints[MODEL],
            map_location=device,
        )
        .to(device)
        .eval()
    )
    generate = lambda mask: model.generate(mask, temperature=1.0, top_p=0.95)[0].argmax(
        axis=1
    )
elif MODEL == "slm_drum":
    model = (
        EncoderOnlyModel.load_from_checkpoint(
            ROOT_DIR + "checkpoints/ruby-surf-331/epoch=1119-step=17920-val/loss_epoch=0.03922.ckpt",
            map_location=device,
        )
        .to(device)
        .eval()
    )
    generate = lambda mask: model.generate(mask, temperature=1.0, top_p=0.95)[0].argmax(
        axis=1
    )

else:
    model = SimplexDiffusionModel.load_from_checkpoint(
        "../checkpoints/dark-sky-67/last.ckpt", map_location=device
    )
    # model = SimplexDiffusionModel.load_from_checkpoint(
    #     "../checkpoints/flowing-paper-64/last.ckpt", map_location=device
    # )

    generate = lambda x: model.sample2(
        mask,
        enforce_prior=True,
        nb_steps=100,
        top_p=0.99,
        batch_size=1,
        prior_strength=1,
        attribute_temperature={
            "pitch": 0.75,
            "onset/tick": 1.0,
            "onset/beat": 1.0,
            "velocity": 1.5,
        },
    )[0]


def preview(sm):
    sm = sm.copy()
    sm = sm_fix_overlap_notes(sm)
    preview_sm(loop_sm(sm, 4, 4))
    # add save button in ipython
    # if pressed a text box will appear to enter the name of the file
    # save the file in the output directory
#%%
# create 128 bpm rock loop with drums, bass, guitar with max 280 notes and minimum 2 drum notes and maximum 40 drum notes

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

# create 128 bpm rock loop with drums, bass, guitar with max 280 notes
n_events = model.tokenizer.config["max_notes"]

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
    def __init__(self):
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
    
    def intersect(self,constraint):    
        for key in constraint:
            self.a[key] = self.a[key] & constraint[key]
        for key in self.a:
            assert  len(self.a[key])>0, f"Empty set for key {key}, constraint {constraint}, attributes {self.a}"
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

scale = get_scale("C pentatonic",range=(30, 100))

def scale_constraint(scale,pitch_range):
    scale_constraint = {
        "pitch": {str(note) for note in get_scale(scale, pitch_range)}
        | {"-"}
    }
    return scale_constraint

#%%

# genre mix
def basic_arrangement():
    e = []
    # 10 forced drums
    # e += [
    #     EventConstraint().intersect({"instrument": {"Drums"}, "tag": {"jazz"}}).force_active()
    #     for _ in range(30)
    # ]
    # 10 forced bass
    # e += [
    #     EventConstraint().intersect({"instrument": {"Drums"}, "pitch":DRUM_PITCHES
    #                                  }).force_active()
    #     for _ in range(30)
    # ]
    # 10 forced piano
    # e += [
    #     EventConstraint()
    #     .intersect(
    #         {
    #             "instrument": {"Bass"},
    #             "pitch": scale_constraint("C major", (30, 50))["pitch"],
    #         }
    #     )
    #     .force_active()
    #     for _ in range(10)
    # ]

    e += [
        EventConstraint()
        .intersect(
            {
                "instrument": {"Piano"},
                # "pitch": scale_constraint("C major", (40, 100))["pitch"],
            }
        )
        .force_active()
        for _ in range(50)
    ]

    # # add bass pitch at 36
    # e += [
    #     EventConstraint().intersect({"instrument": {"Bass"}, "pitch": {"36"}, "onset/beat":{"0"}, "offset/beat":{"1","2","3"}}).force_active()
    # ]
    
    # # 150 optional drums, bass, guitar
    # e += [
    #     EventConstraint().intersect({"instrument": {"Drums","Bass","Piano","Guitar","-"}}) for _ in range(50)
    # ]

    e += [EventConstraint().force_inactive() for _ in range(n_events - len(e))]
    # set tempo and tag
    e = [ev.intersect({"tempo": {"138", "-"},"tag":{"harp","-"}}) for ev in e]
    return e


e = basic_arrangement()


mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = generate(mask)
x_sm = model.tokenizer.decode(x)
preview(x_sm)


#%%

def basic_pop_arrangement():
    e=[]
    # 10 forced drums
    e +=  [EventConstraint().intersect({"instrument":{"Drums"}}).force_active() for _ in range(30)]
    # 10 forced bass
    e += [
        EventConstraint().intersect({"instrument": {"Bass"}}).force_active()
        for _ in range(10)
    ]
    # 10 forced piano
    e += [
        EventConstraint().intersect({"instrument": {"Piano"}}).force_active()
        for _ in range(10)
    ]
    # 10 forced guitar
    e += [
        EventConstraint().intersect({"instrument": {"Guitar"}}).force_active()
        for _ in range(10)
    ]
    # 150 optional drums, bass, guitar
    e += [EventConstraint().intersect({"instrument":{"Drums","Bass","Piano","Guitar","-"}}) for _ in range(150)]
    e += [EventConstraint().force_inactive() for _ in range(n_events - len(e))]
    # set tempo and tag
    e = [ev.intersect({"tempo": {"126","-"},"tag": {"jazz","-"}
                    }) for ev in e]
    return e

e = basic_pop_arrangement()

mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
x = model.generate(mask, temperature=1.0, attribute_temperature={"onset/tick": 1.2,"velocity":1.5})[0].argmax(axis=1)
# x = model.generate(mask, temperature=1.0)[0].argmax(axis=1)
x_sm = model.tokenizer.decode(x)
preview(x_sm)

#%%

# write a jazz arrangement with drums, bass, piano, guitar


def jazz_arrangement():

    # add 10 drums
    e = [EventConstraint().intersect({"instrument": {"Drums"}}).force_active() for _ in range(10)]

    # add 10 bass
    e += [EventConstraint().intersect({"instrument": {"Bass"}}).force_active() for _ in range(10)]

    # add 10 piano
    e += [EventConstraint().intersect({"instrument": {"Piano"}}).force_active() for _ in range(10)]

    # add 10 guitar
    e += [EventConstraint().intersect({"instrument": {"Guitar"}}).force_active() for _ in range(10)]

    # add 150 optional drums, bass, guitar
    e += [EventConstraint().intersect({"instrument": {"Drums","Bass","Piano","Guitar","-"}}) for _ in range(150)]

    # add 10 optional drums
    e += [EventConstraint().intersect({"instrument": {"Drums","-"}}) for _ in range(10)]

    # add 10 optional bass
    e += [EventConstraint().intersect({"instrument": {"Bass","-"}}) for _ in range(10)]


#%%

def create_ghost_notes(e, min=3, max=20):
    # ghost notes to add
    e = [
        EventConstraint()
        .intersect(
            {
                "instrument": {"Drums", "-"},
                "velocity": {"128", "-"},
                "onset/tick": {"3","6","-"},
                "pitch": {"42 (Drums)", "-"},
            }
        )
        .force_active()
        for _ in range(10)
    ]
    e = [
        EventConstraint()
        .intersect(
            {
                "instrument": {"Drums", "-"},
                "velocity": {"128", "-"},
                "onset/tick": {"9","12","-"},
                "pitch": {"42 (Drums)", "-"},
            }
        )
        .force_active()
        for _ in range(10)
    ]
    e += [
        EventConstraint()
        .intersect(
            {
                "instrument": {"Drums", "-"},
                "velocity": {"128", "-"},
                "pitch": {"46 (Drums)", "-"},
            }
        )
        .force_active()
        for _ in range(min)
    ]
    e += [
        EventConstraint().intersect({"instrument": {"Drums","-"}, "velocity": {"50","-"}, "pitch": {"42 (Drums)","-"}})
        for _ in range(max - min)
    ]
    return e

e = sm_to_events(x_sm)

# remove empty events
e = [event for event in e if not event.is_inactive()]

# add ghost notes
e += create_ghost_notes(e, min=5, max=20)

# pad with empty events
e += [EventConstraint().force_inactive() for _ in range(n_events - len(e))]

mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
# x = model.generate(mask, temperature=0.95)[0].argmax(axis=1)
x = generate(mask)
x_sm = model.tokenizer.decode(x)
x_sm = sm_fix_overlap_notes(x_sm)
preview_sm(loop_sm(x_sm,4,4))

#%%

# add tom fill

def add_tom_fill(e):
    # remove drums in last 4 beats
    e = [e for e in e if not (e.a["instrument"] == {"Drums"} and e.a["onset/beat"] & {"12","13","14","15"})]
    # add 3 toms
    e += [EventConstraint().intersect({"instrument": {"Drums","-"},"pitch": {"48 (Drums)","-"}, "onset/beat": {"12","13","14","15","-"}}).force_active() for _ in range(3)]

    # add optional notes
    e += [
        EventConstraint().intersect(
            {"instrument": {"Drums", "-"}, "onset/beat": {"12", "13", "14", "15", "-"}}
        )
        for _ in range(30)
    ]
    return e

e = sm_to_events(x_sm)

# remove empty events
e = [event for event in e if not event.is_inactive()]

# add tom fill
e = add_tom_fill(e)

# pad with empty events
e += [EventConstraint().force_inactive() for _ in range(n_events - len(e))]

mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
# x = model.generate(mask, temperature=1.0)[0].argmax(axis=1)
x = generate(mask)
x_sm2 = model.tokenizer.decode(x)
x_sm2 = sm_fix_overlap_notes(x_sm2)
preview_sm(loop_sm(x_sm2,4,4))



#%%
e = sm_to_events(x_sm)

infill_beat_range = (12, 16)

def infill_beat_range_fn(event, infill_beat_range):
    infill_beats = set([str(r) for r in range(infill_beat_range[0], infill_beat_range[1])])
    if (event.a["onset/beat"] & infill_beats and event.a["offset/beat"] & infill_beats) or event.is_inactive():
        return EventConstraint().intersect(
            {
            "onset/beat": infill_beats | {"-"},
            "offset/beat": infill_beats | {"-"},
            "onset/tick": ALL_ONSET_TICKS | {"-"},
            "offset/tick": ALL_OFFSET_TICKS | {"-"}
            })
    return event

# remove empty events
e = [event for event in e if not event.is_inactive()]

instrument_to_remove = "Guitar"
instrument_to_add = "Guitar"
# remove drums
e = [e for e in e if e.a["instrument"] != {instrument_to_remove}]
# add new drums

e += [EventConstraint().intersect({"instrument": {instrument_to_add}, 
                                #    "pitch": {str(p) for p in range(40, 100)}
                                   }).force_active() for _ in range(40)]
# add optional drums
e += [EventConstraint().intersect({"instrument": {instrument_to_add,"-"}}) for _ in range(30)]

# e = [ev.intersect({"tempo": {"128", "-"}, "tag": {"other", "-"}}) for ev in e]

e += [EventConstraint().force_inactive() for _ in range(n_events - len(e))]
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
# x = model.generate(mask, temperature=0.99)[0].argmax(axis=1)
x = generate(mask)
x_sm = model.tokenizer.decode(x)
x_sm = sm_fix_overlap_notes(x_sm)
preview_sm(loop_sm(x_sm,4,4))

#%% redo dynamics

e = sm_to_events(x_sm)

# remove empty events
e = [event for event in e if not event.is_inactive()]

# blank out all dynamics
for event in e:
    event.a["velocity"] = {"25","50","128"}

# pad with empty events
e += [EventConstraint().force_inactive() for _ in range(n_events - len(e))]

mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)

# x = model.generate(mask, temperature=1.5)[0].argmax(axis=1)
x = generate(mask)
x_sm = model.tokenizer.decode(x)
x_sm = sm_fix_overlap_notes(x_sm)
preview_sm(loop_sm(x_sm, 4, 4))





#%% 

# add piano chords
e = sm_to_events(x_sm)
# remove empty
e = [e for e in e if not e.is_inactive()]


def add_piano_chords(e):
    # remove piano 
    instrument_to_remove = "Piano"
    # remove empty
    e = [e for e in e if e.a["instrument"] != {instrument_to_remove}]

    # add piano chords at the beginning
    e += [EventConstraint().intersect({"instrument": {"Piano"}, "pitch": {str(p) for p in range(50, 100)}, "onset/beat": {"0"}, "offset/beat":{"2","3","4"}}).force_active() for _ in range(6)]

    e += [EventConstraint().intersect({"instrument": {"Piano"}, "pitch": {str(p) for p in range(50, 100)}}).force_active() for _ in range(8)]
    # add up to 16 optional piano notes
    e += [EventConstraint().intersect({"instrument": {"Piano","-"}}) for _ in range(16)]

    return e

e = add_piano_chords(e)

# pad with empty events
e += [EventConstraint().force_inactive() for _ in range(n_events - len(e))]



e += [EventConstraint().force_inactive() for _ in range(n_events - len(e))]
mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)
# x = model.generate(mask, temperature=1.0)[0].argmax(axis=1)
x = generate(mask)
x_sm = model.tokenizer.decode(x)
x_sm = sm_fix_overlap_notes(x_sm)
preview_sm(loop_sm(x_sm,4,4))

# 

#%%



instrument = "Drums"
# write a piano piece with dyamics

# create 20 active piano notes with velocity 25
e = [EventConstraint().intersect({"instrument": {instrument}, "velocity": {"25"}}).force_active() for _ in range(10)]

# create 20 active piano notes with velocity 50
e += [EventConstraint().intersect({"instrument": {instrument}, "velocity": {"50"}}).force_active() for _ in range(10)]

# pad with piano notes
e += [EventConstraint().intersect({"instrument": {instrument, "-"}}) for _ in range(n_events - len(e))]

# constrain everything to cmajor

# e = [ev.intersect(scale_constraint("C pentatonic", (30, 100))) for ev in e]
      
# e = [ev.intersect({"tempo": {"126","-"},"tag": {"classical","-"}}) for ev in e]

mask = model.tokenizer.create_mask([ev.to_dict() for ev in e]).to(device)

x = model.generate(mask, temperature=1.0)[0].argmax(axis=1)
                                  
x_sm = model.tokenizer.decode(x)
x_sm = sm_fix_overlap_notes(x_sm)

preview_sm(x_sm)




#%%

# add drum fill
e = sm_to_events(x_sm)

# fill bars
fill_beat_range = (12, 16)
fill_beats = set([str(r) for r in range(fill_beat_range[0], fill_beat_range[1])])
# remove drums from end
e = [e for e in e if not (e.a["instrument"] != {"Drums"} and e.a["onset/beat"] & fill_beats)]

# add drums to fill




# force all events to have drums
#%%

def apply_constraint(constraint_a, new_constraint):
    for key in new_constraint:
        constraint_a[key] = constraint_a[key] & new_constraint[key]
    # assert that no set is empty
    for key in constraint_a:
        assert len(constraint_a[key]) > 0, f"Empty set for key {key}"
    return constraint_a


def create_dead_events():
    events = [
        {
            "instrument": {"-"},
            "pitch": {"-"},
            "onset/beat": {"-"},
            "onset/tick": {"-"},
            "offset/tick": {"-"},
            "offset/beat": {"-"},
            "velocity": {"-"},
            "tag": {"-"},
            "tempo": {"-"},
        } for _ in range(n_events)
    ]
    return events

def is_dead(event):
    for key in event:
        if event[key] != {"-"}:
            return False
    return True

def transform_rejoin(data, filter_fn, map_fn, limit):
    transformed_events = []
    non_transformed_events = []
    idx = 0
    while idx < len(data):
        event = data[idx]
        if len(transformed_events) < limit and filter_fn(event):
            transformed_event = map_fn(event)
            transformed_events.append(transformed_event)
        else:
            non_transformed_events.append(event)
        idx += 1
    return transformed_events + non_transformed_events

def at_least_n_of_x(events, prototype, n):
    # 
    # filter_fn = lambda x: [attribute in x[key] for key, attribute in prototype.items()].all()
    def respects_prototype(event, prototype):
        for key in prototype.keys():
            # check if intersection is not empty
            if not prototype[key] & event[key]:
                return False
        return True
    
    def overwrite_from_prototype(event, prototype):
        for key, attribute in event.items():
            if key in prototype:
                event[key] = prototype[key]
            else:
                attr = event[key]
                # remove "-"
                attr.discard("-")
                event[key] = attribute
        return event
    return transform_rejoin(events, lambda x: respects_prototype(x, prototype), lambda x: overwrite_from_prototype(x, prototype), n)

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
    return events

def replace_instrument(events, instrument_to_replace, new_instrument, min_notes_per_instrument):
    # remove all events with instrument_to_replace
    new_events = [event for event in events if instrument_to_replace not in event["instrument"]]
    # count     
    
#%%
# create 128 bpm rock loop with drums, bass, guitar with max 280 notes
e = create_dead_events()
filter_fn = lambda x: True
map_fn = lambda x: {
    "instrument": {"Drums", "Bass", "Guitar", "-"},
    "pitch": ALL_PITCH | {"-"},
    "onset/beat": ALL_ONSET_BEATS | {"-"},
    "onset/tick": ALL_ONSET_TICKS | {"-"},
    "offset/beat": ALL_OFFSET_BEATS | {"-"},
    "offset/tick": ALL_OFFSET_TICKS | {"-"},
    "velocity": ALL_VELOCITIES | {"-"},
    "tag": {"rock", "-"},
    "tempo": {"126", "-"},
}

e = transform_rejoin(e, filter_fn, map_fn, 280)
e = at_least_n_of_x(e, {"instrument": {"Drums"}}, 5)
e = at_least_n_of_x(e, {"instrument": {"Bass"}}, 5)
e = at_least_n_of_x(e, {"instrument": {"Guitar"}}, 5)

mask = model.tokenizer.create_mask(e).to(device)

x = model.generate(mask, temperature=0.98)[0].argmax(axis=1)
x_sm = model.tokenizer.decode(x)
x_sm = sm_fix_overlap_notes(x_sm)
preview_sm(x_sm)

#%%



# create a list o


# %%
