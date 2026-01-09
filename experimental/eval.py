import argparse
import os
import glob
import sys
import re
from pathlib import Path
import json
import symusic
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.processors import AudioBoxAesRewardProcessor, TinySoundfontSynthProcessor

def sm_seconds(sm):
    """Calculate duration of a symusic Score in seconds"""
    if len(sm.tempos) == 0:
        tempo = 120  # default tempo
    else:
        tempo = sm.tempos[-1].qpm
    
    ticks = sm.end()
    tpq = sm.ticks_per_quarter
    beats = ticks / tpq
    seconds = beats * 60 / tempo
    return seconds

def load_midi_files(directory):
    """Load all MIDI files from a directory"""
    midi_pattern = os.path.join(directory, "**/*.mid")
    midi_files = glob.glob(midi_pattern, recursive=True)
    return sorted(midi_files)

def parse_system_from_filename(filename):
    """Extract system type from filename (e.g., mlm, slm_cfg0.0, slm_cfg0.5)"""
    basename = os.path.basename(filename).replace('.mid', '')
    
    # Check for MLM files (end with _mlm)
    if basename.endswith('_mlm') or basename.startswith('mlm_'):
        return 'mlm'
    # Check for SLM files with cfg values
    elif 'slm' in basename:
        # Look for cfg pattern like cfg_0.0, cfg_0.5, cfg_1.0, etc.
        cfg_match = re.search(r'cfg_([\d.]+)', basename)
        if cfg_match:
            cfg_val = cfg_match.group(1)
            return f'slm_cfg{cfg_val}'
        elif 'no_cfg' in basename:
            return 'slm_cfg0.0'
        else:
            return 'slm_unknown'
    # Check for source or other files
    elif basename == 'source':
        return 'source'
    else:
        return 'unknown'

def create_records_from_midis(midi_files, max_duration=30):
    """Convert MIDI files to records format expected by processors"""
    records = []
    for i, midi_path in enumerate(tqdm(midi_files, desc="Loading MIDIs")):
        try:
            sm = symusic.Score(midi_path)
            duration = sm_seconds(sm)
            duration = min(duration, max_duration)
            
            record = {
                "idx": i,
                "file_path": midi_path,
                "system": parse_system_from_filename(midi_path),
                "sm": sm,
                "sm_duration": duration,
                "audio": None,
                "sample_rate": None,
                "audio_duration": None,
                "normalized_rewards": {}
            }
            records.append(record)
        except Exception as e:
            print(f"Error loading {midi_path}: {e}")
    
    return records

def eval_directory(directory, soundfont_path=None, sample_rate=44100, max_duration=30):
    """Evaluate all MIDI files in a directory using audiobox aesthetics"""
    
    # Find MIDI files
    midi_files = load_midi_files(directory)
    if not midi_files:
        print(f"No MIDI files found in {directory}")
        return
    
    print(f"Found {len(midi_files)} MIDI files")
    
    # Create records
    records = create_records_from_midis(midi_files, max_duration=max_duration)
    
    # Initialize processors
    print("Initializing synthesizer...")
    if soundfont_path is None:
        # Try to find the soundfont in assets
        assets_dir = Path(__file__).parent.parent / "assets"
        soundfont_path = assets_dir / "MatrixSF_v2.1.5.sf2"
        if not soundfont_path.exists():
            print(f"Soundfont not found at {soundfont_path}")
            print("Please provide a soundfont path with --soundfont")
            return
    
    synth_processor = TinySoundfontSynthProcessor(
        soundfont_path=str(soundfont_path),
        sample_rate=sample_rate,
        max_duration_seconds=max_duration
    )
    
    print("Synthesizing audio...")
    records = synth_processor(records)
    
    print("Initializing audiobox aesthetics predictor...")
    aes_processor = AudioBoxAesRewardProcessor()
    
    print("Computing aesthetics scores...")
    records = aes_processor(records)
    
    # Organize results by system
    results = []
    system_groups = {}
    
    for record in records:
        filename = os.path.basename(record["file_path"])
        scores = record.get("aes_scores", {})
        system = record["system"]
        
        result = {
            "file": filename,
            "system": system,
            "CE": scores.get("CE", None),
            "CU": scores.get("CU", None),
            "PC": scores.get("PC", None),
            "PQ": scores.get("PQ", None),
        }
        results.append(result)
        
        # Group by system
        if system not in system_groups:
            system_groups[system] = []
        system_groups[system].append(result)
    
    # Compute system averages
    system_averages = {}
    for system in sorted(system_groups.keys()):
        system_results = system_groups[system]
        system_avg = {
            "CE": np.mean([r["CE"] for r in system_results if r["CE"] is not None]),
            "CU": np.mean([r["CU"] for r in system_results if r["CU"] is not None]),
            "PC": np.mean([r["PC"] for r in system_results if r["PC"] is not None]),
            "PQ": np.mean([r["PQ"] for r in system_results if r["PQ"] is not None]),
        }
        system_averages[system] = system_avg
    
    # Print table with systems as rows and aspects as columns
    print("\n" + "="*80)
    print("Average Scores by System:")
    print("="*80)
    print(f"{'System':<20} {'CE':>8} {'CU':>8} {'PC':>8} {'PQ':>8}")
    print("-" * 80)
    
    for system in sorted(system_averages.keys()):
        avg = system_averages[system]
        print(f"{system:<20} {avg['CE']:>8.3f} {avg['CU']:>8.3f} {avg['PC']:>8.3f} {avg['PQ']:>8.3f}")
    
    # Compute overall averages
    overall_avg = {
        "CE": np.mean([r["CE"] for r in results if r["CE"] is not None]),
        "CU": np.mean([r["CU"] for r in results if r["CU"] is not None]),
        "PC": np.mean([r["PC"] for r in results if r["PC"] is not None]),
        "PQ": np.mean([r["PQ"] for r in results if r["PQ"] is not None]),
    }
    
    print("\n" + "="*80)
    print("Overall Average Scores:")
    print("="*80)
    print(f"  CE: {overall_avg['CE']:.3f}")
    print(f"  CU: {overall_avg['CU']:.3f}")
    print(f"  PC: {overall_avg['PC']:.3f}")
    print(f"  PQ: {overall_avg['PQ']:.3f}")
    
    # Save results
    output_file = os.path.join(directory, "aesthetics_scores.json")
    with open(output_file, 'w') as f:
        json.dump({
            "results": results,
            "system_averages": system_averages,
            "overall_average": overall_avg
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MIDI files using audiobox aesthetics")
    parser.add_argument("directory", type=str, help="Directory containing MIDI files")
    parser.add_argument("--soundfont", type=str, default=None, help="Path to soundfont file")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    parser.add_argument("--max-duration", type=int, default=30, help="Maximum audio duration in seconds")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Directory not found: {args.directory}")
        sys.exit(1)
    
    eval_directory(args.directory, args.soundfont, args.sample_rate, args.max_duration)
