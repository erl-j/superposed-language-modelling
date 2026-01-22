"""
Generate paper examples demonstrating constraints with MLM and SLM models.
"""

import os
import argparse
import datetime
import torch
import fractions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import symusic

torch.serialization.add_safe_globals([fractions.Fraction])

from slm.load_model_util import load_model
from slm.constraints.core import MusicalEventConstraint
from slm.constraints.re import replace, retime
from slm.constraints.templates import unconditional, pentatonic_guitar, breakbeat, extreme_drums, strings_and_flute, triplet_drums
from slm.constraints.addx import add_percussion, add_snare_ghost_notes, add_tom_fill, add_locked_in_bassline
from slm.util import sm_fix_overlap_notes, plot_piano_roll
from slm.conversion_utils import sm_to_events

# Config
SLM_CFG_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]



def generate(model, events, name, model_name, args, cfg_value=1.0, highlight=None, fixed_pitch_range=None, drums_only=False, locked_sm=None):
    """Core generation function."""
    tokenizer = model.tokenizer
    mask = tokenizer.event_constraints_to_mask(events).to(args.device)
    cfg = cfg_value if model_name != "mlm" else 1.0
    
    # Build filename: constraint_model_cfg
    model_short = "mlm" if model_name == "mlm" else "slm"
    if cfg_value == 1.0 and model_name == "mlm":
        filename = f"{name}_{model_short}"
    else:
        filename = f"{name}_{model_short}_cfg_{cfg_value}"
    
    print(f"  Generating {filename} with {model_name} (CFG={cfg})...")
    out = model.generate(
        mask, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        tokens_per_step=args.tokens_per_step, order="random", constraint_cfg=cfg,
    )
    
    sm = tokenizer.decode(out[0].argmax(-1))
    # sm = sm_fix_overlap_notes(sm)
    
    # Save MIDI
    midi_path = os.path.join(args.midi_dir, f"{filename}.mid")
    sm.dump_midi(midi_path)
    print(f"    Saved: {midi_path}")
    
    # Plot and save directly as PDF
    try:
        fig = plot_piano_roll(sm, highlight=highlight, fixed_pitch_range=fixed_pitch_range, drums_only=drums_only, locked_sm=locked_sm)
        plot_path = os.path.join(args.plot_dir, f"{filename}.pdf")
        fig.savefig(plot_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"    Saved plot: {plot_path}")
    except Exception as e:
        print(f"    Warning: plot failed: {e}")
        import traceback
        traceback.print_exc()
    
    return sm


def build_box_events(tokenizer, source_path, constraint_fn, tick_range, pitch_range, drums, tag, tempo):
    """Build events for box infill constraints."""
    source_sm = symusic.Score(source_path)
    source_events = sm_to_events(source_sm, tag=tag, tokenizer=tokenizer)
    ec = lambda: MusicalEventConstraint(tokenizer)
    tempo = int(source_sm.tempos[0].qpm) if source_sm.tempos else tempo
    return constraint_fn(
        e=source_events, ec=ec, n_events=tokenizer.config["max_notes"],
        tick_range=tick_range, pitch_range=pitch_range, drums=drums, tag=tag, tempo=tempo,
    )


def run_variants(models, generate_fn, name, **kwargs):
    """Run MLM and SLM variants with different CFG values."""
    if "mlm" in models:
        print(f"\n  Generating {name}_mlm")
        generate_fn(models["mlm"], name, "mlm", cfg_value=1.0, **kwargs)
    if "slm" in models:
        for cfg in SLM_CFG_VALUES:
            print(f"\n  Generating {name}_slm_cfg_{cfg}")
            generate_fn(models["slm"], name, "slm_mixed", cfg_value=cfg, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="artefacts/paper_examples")
    parser.add_argument("--epochs", type=int, default=150, choices=[50, 100, 150])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--tokens-per-step", type=int, default=1)
    parser.add_argument("--skip-mlm", action="store_true")
    parser.add_argument("--skip-slm", action="store_true")
    args = parser.parse_args()
    
    # Setup output dirs
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)
    args.midi_dir = os.path.join(args.output_dir, "midi")
    args.plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(args.midi_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Generating Paper Examples")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}, Temp: {args.temperature}, Top-p: {args.top_p}")
    print("=" * 60)
    
    # Load models
    models = {}
    if not args.skip_mlm:
        print("Loading MLM...")
        models["mlm"] = load_model("mlm", epochs=args.epochs, device=args.device)
    if not args.skip_slm:
        print("Loading SLM-Mixed...")
        models["slm"] = load_model("slm_mixed", epochs=args.epochs, device=args.device)
    
    if not models:
        print("No models loaded!")
        return
    
    # Source MIDI
    source_midi = os.path.join(os.path.dirname(__file__), "..", "assets", "drums_bass_piano.mid")
    source_sm = symusic.Score(source_midi)
    source_sm.dump_midi(os.path.join(args.midi_dir, "source.mid"))
    
    # Plot source MIDI
    print("\nPlotting source loop...")
    try:
        fig = plot_piano_roll(source_sm, fixed_pitch_range=(24, 84))
        plot_path = os.path.join(args.plot_dir, "source.pdf")
        fig.savefig(plot_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved plot: {plot_path}")
    except Exception as e:
        print(f"  Warning: source plot failed: {e}")
    
    # Standard pitch range for harmonic: C1-C6
    HARMONIC_PITCH_RANGE = (24, 84)
    
    # Box infill examples
    box_examples = [
        ("replace_harmonic_center", replace, [96, 288], [30, 74]),
        ("retime_full_piece", retime, [0, 384], [0, 128]),
    ]
    
    for name, constraint_fn, tick_range, pitch_range in box_examples:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        
        highlight = {"tick_range": tick_range, "pitch_range": pitch_range}
        
        def gen_fn(model, name, model_name, cfg_value, constraint_fn=constraint_fn, tick_range=tick_range, pitch_range=pitch_range, **_):
            tokenizer = model.tokenizer
            events = build_box_events(
                tokenizer, source_midi, constraint_fn, tick_range, pitch_range, 
                drums=False, tag="pop", tempo=120
            )
            generate(model, events, name, model_name, args, cfg_value, highlight, HARMONIC_PITCH_RANGE)
        
        run_variants(models, gen_fn, name)
    
    # Generation from scratch examples
    scratch_examples = [
        ("unconditional", unconditional, False),
        ("pentatonic_guitar", pentatonic_guitar, False),
        ("strings_and_flute", strings_and_flute, False),
        ("breakbeat", breakbeat, False),
        ("extreme_drums", extreme_drums, False),
        ("triplet_drums", triplet_drums, False),
    ]
    
    for name, constraint_fn, drums_only in scratch_examples:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        
        def gen_fn(model, name, model_name, cfg_value, constraint_fn=constraint_fn, drums_only=drums_only, **_):
            tokenizer = model.tokenizer
            ec = lambda: MusicalEventConstraint(tokenizer)
            events = constraint_fn(
                e=[], ec=ec, n_events=tokenizer.config["max_notes"],
                beat_range=[0, 16], pitch_range=[0, 128], drums=drums_only,
                tag="pop", tempo=120,
            )
            generate(model, events, name, model_name, args, cfg_value, 
                    fixed_pitch_range=None if drums_only else HARMONIC_PITCH_RANGE,
                    drums_only=drums_only)
        
        run_variants(models, gen_fn, name)
    
    # Additive edit examples
    additive_examples = [
        ("add_percussion", add_percussion, False),
        ("add_snare_ghost_notes", add_snare_ghost_notes, False),
        ("add_tom_fill", add_tom_fill, False),
        ("add_locked_in_bassline", add_locked_in_bassline, False),
    ]
    
    for name, constraint_fn, drums_only in additive_examples:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        
        def gen_fn(model, name, model_name, cfg_value, constraint_fn=constraint_fn, drums_only=drums_only, **_):
            tokenizer = model.tokenizer
            source_events = sm_to_events(source_sm, tag="pop", tokenizer=tokenizer)
            ec = lambda: MusicalEventConstraint(tokenizer)
            events = constraint_fn(
                e=source_events, ec=ec, n_events=tokenizer.config["max_notes"],
                beat_range=[0, 16], pitch_range=[0, 128], drums=drums_only,
                tag="pop", tempo=120,
            )
            # Pass source_sm as locked_sm to visualize what was locked
            generate(model, events, name, model_name, args, cfg_value,
                    fixed_pitch_range=None if drums_only else HARMONIC_PITCH_RANGE,
                    drums_only=drums_only,
                    locked_sm=source_sm)
        
        run_variants(models, gen_fn, name)
    
    print(f"\n{'=' * 60}\nDone! Output: {args.output_dir}\n{'=' * 60}")


if __name__ == "__main__":
    main()