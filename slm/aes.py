#!/usr/bin/env python3
# MIDI to Audio Conversion Script
#
# This script:
# 1. Renders MIDI files to audio using a soundfont
# 2. Tracks rendered audio in a JSONL file
# 3. Processes the audio with audio-aes
# 4. Combines the original records with AES output into a CSV

import os
import json
import glob
from tqdm import tqdm
import pandas as pd
import soundfile as sf
import argparse
from symusic import Score, Synthesizer, BuiltInSF3

# Parse command line arguments
parser = argparse.ArgumentParser(description='Convert MIDI files to audio and analyze them.')
parser.add_argument('--input-dir', type=str, default='./artefacts/applications_250e/ground_truth/', 
                    help='Directory containing MIDI files')
parser.add_argument('--output-dir', type=str, default=None, 
                    help='Directory for output audio files (defaults to input_dir with "audio" appended)')
parser.add_argument('--cuda-device', type=int, default=0, 
                    help='CUDA device index for audio-aes')
parser.add_argument('--sample-rate', type=int, default=44100, 
                    help='Sample rate for audio conversion')
parser.add_argument('--batch-size', type=int, default=10, 
                    help='Batch size for audio-aes')
args = parser.parse_args()

# Setup directories
input_dir = args.input_dir
# If output_dir is not specified, append "audio" to input_dir
if args.output_dir is None:
    # Handle trailing slashes properly
    input_dir_clean = input_dir.rstrip('/')
    output_dir = f"{input_dir_clean}_audio"
else:
    output_dir = args.output_dir
sample_rate = args.sample_rate

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")

# Setup soundfont for synthesis
# Use the default MuseScore General soundfont
sf_path = BuiltInSF3.MuseScoreGeneral().path(download=True)

# Create synthesizer with the specified sample rate
synth = Synthesizer(sf_path=sf_path, sample_rate=sample_rate)

# Find all MIDI files
midi_paths = glob.glob(f"{input_dir}/**/*.mid", recursive=True)
print(f"Found {len(midi_paths)} MIDI files")

# # Initialize list to store audio metadata records
aba_records = []

# # Convert MIDI files to audio
# for midi_path in tqdm(midi_paths, desc="Converting MIDI to audio"):
#     try:
#         # Load MIDI score
#         score = Score(midi_path)
        
#         # Render audio (stereo by default)
#         audio = synth.render(score, stereo=True)
        
#         # Preserve directory structure by mirroring the input path in the output directory
#         # Get the relative path from the input directory
#         rel_path = os.path.relpath(midi_path, input_dir)
#         # Create the corresponding output path with .wav extension
#         output_path = os.path.join(output_dir, rel_path.replace(".mid", ".wav"))
#         # Create parent directories if they don't exist
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#         # Write audio file
#         sf.write(
#             output_path,
#             audio.T,
#             sample_rate
#         )
        
#         # Create record with audio metadata
#         duration = audio.shape[1] / sample_rate
#         # Store the path that's relative to the current working directory
#         # This ensures the audio-aes tool can find the files
#         rel_to_cwd_path = os.path.relpath(output_path, os.getcwd())
#         aba_records.append({
#             "path": rel_to_cwd_path,
#             "start_time": 0,
#             "end_time": duration
#         })
#     except Exception as e:
#         print(f"Error converting {midi_path}: {e}")

# Save records to JSONL file
records_path = os.path.join(output_dir, "aba_records.jsonl")
# with open(records_path, "w") as f:
#     for record in aba_records:
#         f.write(json.dumps(record) + "\n")

print(f"Saved records to: {records_path}")

# Run audio-aes on the records
scores_path = os.path.join(output_dir, "output.jsonl")
cmd = f"CUDA_VISIBLE_DEVICES={args.cuda_device} audio-aes {records_path} --batch-size {args.batch_size} > {scores_path}"
print(f"Running: {cmd}")
status = os.system(cmd)

# Check if audio-aes ran successfully
if status == 0:
    # Load audio records and scores
    with open(records_path, "r") as f:
        records = [json.loads(line) for line in f.readlines()]
    
    with open(scores_path, "r") as f:
        scores = [json.loads(line) for line in f.readlines()]
    
    # Merge records and scores by index
    merged_records = [{**record, **score} for record, score in zip(records, scores)]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(merged_records)
    csv_path = os.path.join(output_dir, "aba_records.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Process completed successfully. Results saved to: {csv_path}")
else:
    print(f"Error running audio-aes. Exit code: {status}")