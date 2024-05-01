from util import sm_fix_overlap_notes
import symusic
import glob
import argparse
import os
import pretty_midi as pm


def sm_enforce_canonical_track_order(sm):
    sm = sm.copy()
    program_nrs = [track.program for track in sm.tracks if not track.is_drum]

    # set track named piano to program 2
    # for track in sm.tracks:
    #     if track.name == "Piano":
    #         track.program = 2


    # if no drum track exists, add drum track
    if not any([track.is_drum for track in sm.tracks]):
        track = symusic.Track(program = 0, is_drum = True)
        sm.tracks.append(track)

    # sort tracks by is_drum
    sm.tracks = sorted(sm.tracks,key=lambda x: x.is_drum)
    # sort tracks by program
    sm.tracks = sorted(sm.tracks,key=lambda x: x.program)


    program_to_instrument_class = {program_nr:pm.program_to_instrument_class(program_nr) for program_nr in range(128)}

    instrument_class_to_programs = {instrument_class:[] for instrument_class in set(program_to_instrument_class.values())}
    for program_nr, instrument_class in program_to_instrument_class.items():
        instrument_class_to_programs[instrument_class].append(program_nr)   
    
    instrument_classes = list(set(program_to_instrument_class.values()))

    sm_programs = [track.program for track in sm.tracks if not track.is_drum]

    sm_classes = [pm.program_to_instrument_class(program_nr) for program_nr in sm_programs]

    print("Instrument classes:")
    for instrument_class in instrument_classes:
        print(instrument_class)

    print("\n")
    print("SM classes:")
    for sm_class in sm_classes:
        print(sm_class)

    missing_classes = set(instrument_classes) - set(sm_classes)

    print(sm.tracks)
    print("Missing classes:")
    for missing_class in missing_classes:
        print(missing_class)
    for missing_class in missing_classes:
        missing_program = instrument_class_to_programs[missing_class][0]
        track = symusic.Track(program = missing_program, name= pm.program_to_instrument_class(missing_program))
        sm.tracks.append(track)
    
    # sort tracks by is_drum
    sm.tracks = sorted(sm.tracks,key=lambda x: x.is_drum, reverse=True)
    # sort tracks by program
    sm.tracks = sorted(sm.tracks,key=lambda x: x.program)

    print("\n")
    print("Sorted tracks:")
    for track in sm.tracks:
        print(track.name)
    return sm

if __name__ == "__main__":
    # src dir for midi files
    parser = argparse.ArgumentParser()
    # make
    parser.add_argument("--src_midi_path", type=str, required=True)
    parser.add_argument("--dst_midi_path", type=str, required=False)

    args = parser.parse_args()
    src_midi_path = args.src_midi_path
    # if not specified, add _pp to the end of the src_midi_path
    dst_midi_path = args.dst_midi_path if args.dst_midi_path else src_midi_path + "_pp"


    midi_files = glob.glob(src_midi_path + "/**/*.mid", recursive=True)

    for midi_file in midi_files:

        sm = symusic.Score(midi_file)
        sm = sm_fix_overlap_notes(sm)
        # sm = sm_enforce_canonical_track_order(sm)
        out_path = midi_file.replace(src_midi_path, dst_midi_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sm.dump_midi(out_path)


        


    