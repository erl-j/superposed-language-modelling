from .tokenizer import instrument_class_to_selected_program_nr
import symusic
from .constraints.core import MusicalEventConstraint
import einops

def looprep_to_sm(looprep, tpq):
    midi = symusic.Score()
    # set tempo
    midi.tempos = [symusic.Tempo(time=0, qpm=looprep["tempo"])]
    # set time signature
    midi.time_signatures = [symusic.TimeSignature(time=0, numerator=4, denominator=4)]
    midi.tpq = tpq

    # add drum track
    drum_track = symusic.Track(name="Drums", program=0, is_drum=True)
    for note_event in looprep["drum_seq"]:
        drum_track.notes.append(
            symusic.Note(
                pitch=note_event["pitch"],
                time=note_event["onset"],
                duration=note_event["duration"],
                velocity=note_event["velocity"],
            )
        )
    midi.tracks.append(drum_track)

    # add other tracks
    # group by instrument
    instruments = {}
    for note_event in looprep["harm_seq"]:
        if note_event["instrument"] not in instruments:
            instruments[note_event["instrument"]] = symusic.Track(
                name=note_event["instrument"],
                program=instrument_class_to_selected_program_nr[
                    note_event["instrument"]
                ],
                is_drum=False,
            )
        instruments[note_event["instrument"]].notes.append(
            symusic.Note(
                pitch=note_event["pitch"],
                time=note_event["onset"],
                duration=note_event["duration"],
                velocity=note_event["velocity"],
            )
        )
    for instrument, track in instruments.items():
        midi.tracks.append(track)

    return midi


def sm_to_events(x_sm, tag, tokenizer):
    ec = lambda : MusicalEventConstraint(tokenizer)
    x = tokenizer.encode(x_sm, tag=tag, midi_type="loop" if len(tokenizer.config["midi_types"]) > 1 else None)
    if not tokenizer.config["fold_event_attributes"]:
        x = einops.rearrange(x, "event attribute -> (event attribute)")
    tokens = tokenizer.indices_to_tokens(x)
    # group by n_attributes
    n_attributes = len(tokenizer.note_attribute_order)
    events = []
    for i in range(0, len(tokens), n_attributes):
        event = {key: set() for key in tokenizer.note_attribute_order}
        for j in range(n_attributes):
            token = tokens[i + j]
            key, value = token.split(":")
            event[key].add(value)
        events.append(event)
    # create event objects
    events = [ec().intersect(event) for event in events]
    return events


def sm_to_looprep(sm):
    # make two sequences. one for drums and one for other instruments
    n_bars = 4
    drum_sequence = []
    harm_sequence = []
    for track in sm.tracks:
        for note in track.notes:
            if track.name == "Drums":
                drum_sequence.append(
                    {
                        "pitch": note.pitch,
                        "onset": note.start % (sm.tpq * n_bars * 4),
                        "velocity": note.velocity,
                        "duration": sm.tpq // 4,
                        "instrument": track.name,
                    }
                )
            else:
                harm_sequence.append(
                    {
                        "pitch": note.pitch,
                        "onset": note.start % (sm.tpq * n_bars * 4),
                        "velocity": note.velocity,
                        "duration": note.duration,
                        "instrument": track.name,
                    }
                )
    return {
        "time_signature": "4/4",
        "tempo": sm.tempos[0].qpm,
        "n_bars": n_bars,
        "drum_seq": drum_sequence,
        "harm_seq": harm_sequence,
        "ppq": sm.tpq,
    }