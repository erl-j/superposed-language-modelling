import symusic
import glob
# Load a MIDI file

root_path = "artefacts/eval_cropped_midi/fad_test_sane"


file_paths = glob.glob(f"{root_path}/**/*.mid", recursive=True)
# if infill harmonic 

no_drums_tasks = {
    "infilling_harmonic",
    "infilling_box_end",
    "infilling_box_middle",
    "infilling_high",
    "infilling_high_patched",
    "infilling_low",
}
for file_path in file_paths:
    if "natural" not in file_path:
     
        # remove src_path
        cropped_path = file_path.replace(root_path+"/", "")
        
       

        task, system, gen_midi_fn = cropped_path.split("/")

        # load score
        gen_sm = symusic.Score(file_path)

        # load original
        src_path = file_path.replace(task+"/"+system+"/", "natural/")
        src_sm = symusic.Score(src_path)

        if task in no_drums_tasks:
            # replace all drums in gen with src
            src_drum_tracks = [track for track in src_sm.tracks if track.is_drum]
            gen_tracks = [track for track in gen_sm.tracks if not track.is_drum]
            gen_sm.tracks = gen_tracks + src_drum_tracks
            gen_sm.dump_midi(file_path.replace(".mid",".mid"))
            print(f"Sanitized {file_path}")
        elif task == "infilling_drums":
            # replace all non drums in gen with src
            src_non_drum_tracks = [track for track in src_sm.tracks if not track.is_drum]
            gen_tracks = [track for track in gen_sm.tracks if track.is_drum]
            gen_sm.tracks = gen_tracks + src_non_drum_tracks
            gen_sm.dump_midi(file_path.replace(".mid",".mid"))
            print(f"Sanitized {file_path}")


            







        # sanitize midi



