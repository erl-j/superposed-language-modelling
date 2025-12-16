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
from matplotlib.patches import Rectangle
import numpy as np
import pretty_midi
import symusic

torch.serialization.add_safe_globals([fractions.Fraction])

from slm.load_model_util import load_model
from slm.constraints.core import MusicalEventConstraint
from slm.constraints.re import replace, retime
from slm.constraints.templates import unconditional, pentatonic_guitar, breakbeat, extreme_drums, strings_and_flute, triplet_drums
from slm.constraints.addx import add_percussion, add_snare_ghost_notes, add_tom_fill, add_locked_in_bassline
from slm.util import sm_fix_overlap_notes
from slm.conversion_utils import sm_to_events

# Config
SLM_CFG_VALUES = [0.0, 0.5, 1.0, 1.5]
INSTRUMENT_COLORS = {
    "Piano": (0.2, 0.4, 0.8), "Bass": (0.8, 0.2, 0.2), "Guitar": (0.2, 0.7, 0.3),
    "Drums": (0.1, 0.1, 0.1), "Strings": (0.6, 0.3, 0.7), "Brass": (0.9, 0.6, 0.1),
    "Reed": (0.4, 0.7, 0.7), "Organ": (0.5, 0.3, 0.1), "Synth Lead": (0.9, 0.3, 0.6),
    "Synth Pad": (0.4, 0.5, 0.8), "Chromatic Percussion": (0.7, 0.7, 0.2),
}

def plot_piano_roll(sm, highlight=None, fixed_pitch_range=None, drums_only=False, locked_sm=None):
    """
    Plot piano roll with optional highlight box and locked notes visualization.
    locked_sm: A symusic Score containing notes that were constrained/fixed.
    """
    pr_tpq = 12
    tempo = int(sm.tempos[-1].qpm) if sm.tempos else 120
    time_sig = f"{sm.time_signatures[-1].numerator}/{sm.time_signatures[-1].denominator}" if sm.time_signatures else "4/4"
    
    instrument_names = [
        pretty_midi.program_to_instrument_class(t.program) if not t.is_drum else "Drums" 
        for t in sm.tracks
    ]
    
    # Generate piano rolls for the main score
    pr = sm.copy().resample(pr_tpq, min_dur=0).pianoroll(modes=["frame"])[0]
    
    # Generate piano rolls for the locked notes if provided
    locked_pr = None
    if locked_sm:
        try:
            locked_pr = locked_sm.copy().resample(pr_tpq, min_dur=0).pianoroll(modes=["frame"])[0]
        except Exception:
            pass # Fail silently if locked_sm is incompatible or empty

    unique_instruments = np.unique(instrument_names)
    instrument_to_tracks = {inst: [i for i, n in enumerate(instrument_names) if n == inst] for inst in unique_instruments}
    
    loop_ticks = pr_tpq * 4 * 4
    drum_indices = np.where(np.array(instrument_names) == "Drums")[0]
    has_drums = len(drum_indices) > 0
    melodic_indices = [i for i, n in enumerate(instrument_names) if n != "Drums"]
    has_melodic = len(melodic_indices) > 0 and not drums_only
    
    # Determine pitch range
    all_pitches = []
    for idx in melodic_indices:
        if idx < len(pr):
            all_pitches.extend(np.where(np.any(pr[idx][:, :loop_ticks] > 0, axis=1))[0])
    
    if fixed_pitch_range:
        min_pitch, max_pitch = fixed_pitch_range
    elif all_pitches:
        min_pitch, max_pitch = max(0, min(all_pitches) - 3), min(127, max(all_pitches) + 3)
    else:
        min_pitch, max_pitch = 36, 84  # C2 to C6
    
    # Create figure - reduced height for harmonic plots
    if has_drums and has_melodic:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.4, 8), gridspec_kw={'height_ratios': [2, 1]})
        plt.subplots_adjust(hspace=0.3)
    elif has_drums and not has_melodic:
        fig, ax2 = plt.subplots(figsize=(9.4, 4))
        ax1 = None
    else:
        fig, ax1 = plt.subplots(figsize=(8.0, 5))
        ax2 = None
    
    time_ticks = np.arange(0, loop_ticks, pr_tpq)
    
    # Helper to plot notes with per-instrument colors
    def plot_notes_on_ax(ax, track_indices, is_drum, p_roll_source, is_locked_layer=False, drum_pitch_map=None):
        if ax is None: return
        
        fallback_colors = plt.cm.Set2.colors
        target_indices = [idx for idx in track_indices if idx < len(p_roll_source)]
        if not target_indices: return

        if is_drum:
            # Aggregate drums
            combined_pr = sum(p_roll_source[idx] for idx in target_indices)[:, :loop_ticks]
            nz = np.where(combined_pr > 0)
            if len(nz[0]) > 0:
                vel = combined_pr[nz].astype(float)
                
                # Map MIDI pitches to ordinal positions if pitch_map provided
                if drum_pitch_map is not None:
                    y_positions = [drum_pitch_map.get(p, p) for p in nz[0]]
                else:
                    y_positions = nz[0]
                
                if is_locked_layer:
                    # Plot locked drums with hatched rectangles
                    for t, y in zip(nz[1], y_positions):
                        rect = Rectangle((t, y - 0.4), 1, 0.8, 
                                       facecolor='lightgray', edgecolor='gray',
                                       hatch='///', linewidth=0.5, alpha=0.6)
                        ax.add_patch(rect)
                else:
                    # Plot drums as rectangles
                    brightness = np.clip(vel / 127.0, 0.0, 1.0)
                    for t, y, b in zip(nz[1], y_positions, brightness):
                        color = np.clip(np.array([0, 0, 0]) + b, 0.0, 1.0)
                        rect = Rectangle((t, y - 0.4), 1, 0.8, 
                                       facecolor=color, edgecolor='black', linewidth=0.3)
                        ax.add_patch(rect)

        else:
            # Melodic - plot per instrument with colors
            for i, inst in enumerate(unique_instruments):
                if inst == "Drums": continue
                
                inst_indices = [x for x in instrument_to_tracks[inst] if x in target_indices]
                if not inst_indices: continue
                
                inst_pr = p_roll_source[inst_indices].sum(axis=0)[:, :loop_ticks]
                nz = np.where(inst_pr > 0)
                if len(nz[0]) == 0: continue
                
                if is_locked_layer:
                    # Plot locked melodic notes with hatched rectangles
                    for t, p in zip(nz[1], nz[0]):
                        rect = Rectangle((t, p - 0.4), 1, 0.8,
                                       facecolor='lightgray', edgecolor='gray',
                                       hatch='///', linewidth=0.5, alpha=0.6)
                        ax.add_patch(rect)
                else:
                    vel = inst_pr[nz].astype(float)
                    brightness = np.clip((1.0 - vel / 127.0) * 0.5, 0.0, 0.5)
                    color = np.array(INSTRUMENT_COLORS.get(inst, fallback_colors[i % len(fallback_colors)]))[None, :]
                    colors = np.clip(color + (1.0 - color) * brightness[:, None], 0.0, 1.0)
                    ax.scatter(nz[1] + 0.5, nz[0], color=colors, marker='s', s=15, label=inst)

    # Create drum pitch mapping for ordinal y-axis
    drum_pitch_map = None
    unique_drum_pitches = []
    if ax2 and has_drums:
        # Collect all unique drum pitches from main and locked scores
        drum_pr_main = sum(pr[idx] for idx in drum_indices if idx < len(pr))[:, :loop_ticks]
        nz_main = np.where(drum_pr_main > 0)
        unique_pitches = set(nz_main[0])
        
        # Also include locked drums if present
        if locked_pr is not None:
            l_inst_names = [
                pretty_midi.program_to_instrument_class(t.program) if not t.is_drum else "Drums" 
                for t in locked_sm.tracks
            ]
            l_drum_idxs = np.where(np.array(l_inst_names) == "Drums")[0]
            if len(l_drum_idxs) > 0:
                l_drum_pr = sum(locked_pr[idx] for idx in l_drum_idxs if idx < len(locked_pr))[:, :loop_ticks]
                nz_l = np.where(l_drum_pr > 0)
                unique_pitches.update(nz_l[0])
        
        # Sort pitches in drum kit order (low to high is typical)
        unique_drum_pitches = sorted(list(unique_pitches))
        drum_pitch_map = {pitch: i for i, pitch in enumerate(unique_drum_pitches)}
    
    # 1. Plot the Generated (Main) Score
    if ax1:
        plot_notes_on_ax(ax1, melodic_indices, False, pr, is_locked_layer=False)
    if ax2:
        plot_notes_on_ax(ax2, drum_indices, True, pr, is_locked_layer=False, drum_pitch_map=drum_pitch_map)

    # Skip overlaying locked notes with hatching - just show the generated notes

    # Standard Formatting
    if ax1:
        ax1.set_xticks(time_ticks)
        ax1.set_xticklabels(np.arange(len(time_ticks)))
        ax1.set_xlim(0, loop_ticks)
        ax1.set_ylim(min_pitch, max_pitch)
        pitch_ticks = [p for p in range(min_pitch - (min_pitch % 12), max_pitch + 12, 12) if min_pitch <= p <= max_pitch]
        ax1.set_yticks(pitch_ticks)
        ax1.set_yticklabels([f"C{p // 12 - 1}" for p in pitch_ticks])
        ax1.grid(True, which='both', linestyle='--', alpha=0.3, color='gray')
        ax1.set_xlabel("Beat")
        ax1.set_ylabel("Pitch")
        ax1.set_title(f"Melodic Instruments - Tempo: {tempo}, Time Signature: {time_sig}")
        
        # Fix duplicate labels in legend
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax1.legend(by_label.values(), by_label.keys(), title="Instruments", loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Highlight box (Infill)
        if highlight:
            scale = pr_tpq / 24  # tokenizer tpq = 24
            t0, t1 = [t * scale for t in highlight["tick_range"]]
            p0, p1 = highlight["pitch_range"]
            ax1.add_patch(Rectangle((t0, p0), t1 - t0, p1 - p0, lw=1.5, ec="magenta", fc="none", ls="--", alpha=0.8))
    
    if ax2 and has_drums and unique_drum_pitches:
        ordinal_positions = list(range(len(unique_drum_pitches)))
        
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(np.arange(len(time_ticks)))
        ax2.set_xlim(0, loop_ticks)
        # Set y-ticks at ordinal positions with drum names as labels
        ax2.set_yticks(ordinal_positions)
        ax2.set_yticklabels([pretty_midi.note_number_to_drum_name(p) or str(p) for p in unique_drum_pitches])
        ax2.set_ylim(-0.5, len(unique_drum_pitches) - 0.5)
        ax2.grid(True, which='both', linestyle='--', alpha=0.3, color='gray')
        ax2.set_xlabel("Beat")
        ax2.set_ylabel("Drum Type")
        ax2.set_title("Drum Pattern")
    
    plt.tight_layout()
    return fig


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
    sm = sm_fix_overlap_notes(sm)
    
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
    parser.add_argument("--output-dir", default="paper_examples")
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
        fig = plot_piano_roll(source_sm, fixed_pitch_range=(36, 84))
        plot_path = os.path.join(args.plot_dir, "source.pdf")
        fig.savefig(plot_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved plot: {plot_path}")
    except Exception as e:
        print(f"  Warning: source plot failed: {e}")
    
    # Standard pitch range for harmonic: C2-C6
    HARMONIC_PITCH_RANGE = (36, 84)
    
    # Box infill examples
    box_examples = [
        ("replace_harmonic_center", replace, [96, 288], [48, 74]),
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
        ("breakbeat", breakbeat, True),
        ("extreme_drums", extreme_drums, True),
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
        ("add_percussion", add_percussion, True),
        ("add_snare_ghost_notes", add_snare_ghost_notes, True),
        ("add_tom_fill", add_tom_fill, True),
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