#%%
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from slm import load_model
from slm.constraints.core import MusicalEventConstraint

USE_FP16 = False
DEVICE = "cuda:7"

MODEL_TYPES = ["slm_mixed", "slm_sparse"]

# Figure layout constants
FIGURE_SIZE = (20, 18)                 # (width, height) for figures
SUBPLOT_ROW_SPACING = 1.0              # hspace between subplot rows (octave experiment)
SUBPLOT_ROW_SPACING_DRUMS = 0.75       # hspace between subplot rows (drum experiment)
TITLE_PADDING = 2                    # padding above subplot titles
TITLE_FONT_SIZE = 18                 # font size for barplot titles
CONSTRAINT_LABEL_Y_POSITION = 1.2    # y-position of constraint text above subplot (in axes coords)
CONSTRAINT_LABEL_FONT_SIZE = 20       # font size for constraint label text
AXIS_LABEL_FONT_SIZE = 20              # font size for axis labels (xlabel, ylabel)
TICK_LABEL_SIZE = 16                   # font size for tick labels
XTICKLABEL_ROTATION = 45               # rotation angle for x-axis tick labels

# C# major triad (C#, F, G#) at different octaves
# MIDI: C#=1,13,25,37,49,61,73,85,97... F=C#+3, G#=C#+8
OCTAVE_CONDITIONS = {
    "low": {"pitches": [37, 41, 44], "label": "C#2-F2-G#2 (Low)"},    # C#2, F2, G#2
    "mid": {"pitches": [61, 65, 68], "label": "C#4-F4-G#4 (Mid)"},    # C#4, F4, G#4
    "high": {"pitches": [85, 89, 92], "label": "C#6-F6-G#6 (High)"},  # C#6, F6, G#6
}

# Drum pitch conditions (GM drums)
# Crash: 49, 57 | China: 52 | Splash: 55
# Snare: 38, Clap: 39
# Toms: 41, 43, 45, 47, 48, 50
DRUM_PITCH_CONDITIONS = {
    "kick_crash": {
        "pitches": {"49 (Drums)", "52 (Drums)", "55 (Drums)", "57 (Drums)"},
        "label": "Crash 1 & 2 / China / Splash",
        "constraint_text": "pitch ∈ {49, 52, 55, 57} (Crash 1 & 2/China/Splash)",
    },
    "snare_clap": {
        "pitches": {"38 (Drums)", "39 (Drums)"},
        "label": "Snare + Clap",
        "constraint_text": "pitch ∈ {38, 39} (Snare/Hand Clap)",
    },
    "toms": {
        "pitches": {
            "41 (Drums)",
            "43 (Drums)",
            "45 (Drums)",
            "47 (Drums)",
            "48 (Drums)",
            "50 (Drums)",
        },
        "label": "Toms",
        "constraint_text": "pitch ∈ {41, 43, 45, 47, 48, 50} (Toms)",
    },
}

def run_experiments_for_model(model_type):
    print(f"\n{'='*60}")
    print(f"Loading model: {model_type}")
    print(f"{'='*60}")
    
    model = load_model(
        model_type=model_type,
        epochs=150,
        device=DEVICE
    )
    
    if USE_FP16:
        model = model.half()
    
    N_EVENTS = model.tokenizer.config["max_notes"]
    ec = lambda: MusicalEventConstraint(model.tokenizer)
    
    output_dir = f"neighbour_constraints/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for octave_name, octave_info in OCTAVE_CONDITIONS.items():
        pitches = octave_info["pitches"]
        label = octave_info["label"]
        print(f"  Running: {label}")
        
        # Build constraints
        e = []
        # Event A: Constrained to C, E, G at this octave
        e += [ec()
              .intersect({"pitch": {str(p) for p in pitches}})
              .force_active()]
        # Event B: Unconstrained
        e += [ec().force_active() for _ in range(29)]
        # Pad remaining as inactive
        e += [ec().force_inactive() for _ in range(N_EVENTS - len(e))]
        
        mask = model.tokenizer.event_constraints_to_mask(e).to(DEVICE)
        out_logits = model.forward(mask.float())
        
        # Get event B (unconstrained) probabilities
        event_b_logits = out_logits[0,1]
        event_b_probs = torch.softmax(event_b_logits, dim=-1)

        event_a_probs = torch.softmax(out_logits[0,0], dim=-1)

        
        # Instrument distribution (from constrained event A)
        instrument_vocab = [token for token in model.tokenizer.vocab if "instrument:" in token and "Drums" not in token]
        instrument_index = model.tokenizer.note_attribute_order.index("instrument")
        instrument_token_indices = [model.tokenizer.vocab.index(token) for token in instrument_vocab]
        instrument_probs = event_a_probs[instrument_index, instrument_token_indices].cpu().detach().numpy()
        # Renormalize after filtering
        instrument_probs = instrument_probs / instrument_probs.sum()
        
        # Pitch distribution (from unconstrained event B)
        pitch_vocab = [token for token in model.tokenizer.vocab 
                       if "pitch:" in token and "-" not in token and "Drums" not in token
                       and 24 <= int(token.split(":")[1]) <= 108]
        pitch_index = model.tokenizer.note_attribute_order.index("pitch")
        pitch_token_indices = [model.tokenizer.vocab.index(token) for token in pitch_vocab]
        pitch_probs = event_b_probs[pitch_index, pitch_token_indices].cpu().detach().numpy()
        # Renormalize after filtering
        pitch_probs = pitch_probs / pitch_probs.sum()
        
        results[octave_name] = {
            "label": label,
            "pitch_vocab": pitch_vocab,
            "pitch_probs": pitch_probs,
            "instrument_vocab": instrument_vocab,
            "instrument_probs": instrument_probs,
        }
    
    # Create figure with 3 rows (octaves) x 2 cols (pitch, instrument)
    fig, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE)
    plt.subplots_adjust(hspace=SUBPLOT_ROW_SPACING)
    
    for row_idx, octave_name in enumerate(["low", "mid", "high"]):
        r = results[octave_name]
        
        # Add row title showing the constraint (centered above both columns)
        pitches = OCTAVE_CONDITIONS[octave_name]["pitches"]
        pitch_names = [f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][p%12]}{p//12-1}" for p in pitches]
        constraint_text = f"Constraint: pitch ∈ {{{', '.join(pitch_names).lower()}}}"
        # Position text above the left subplot, but centered between both
        axes[row_idx, 0].text(
            1.0,
            CONSTRAINT_LABEL_Y_POSITION,
            constraint_text,
            transform=axes[row_idx, 0].transAxes,
            ha='center',
            va='bottom',
            fontsize=CONSTRAINT_LABEL_FONT_SIZE,
            fontweight='bold',
            color='#2c3e50',
            clip_on=False,
        )
        
        # Instrument distribution (left - constrained event)
        ax_inst = axes[row_idx, 0]
        inst_labels = [v.split(":")[1] for v in r["instrument_vocab"]]
        ax_inst.bar(range(len(r["instrument_probs"])), r["instrument_probs"], color='lightsteelblue')
        ax_inst.set_title("Constrained event, instrument probability distribution", fontsize=TITLE_FONT_SIZE, pad=TITLE_PADDING)
        ax_inst.set_ylabel("Probability", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_inst.set_ylim(0, 0.5)
        ax_inst.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
        # replace "Chromatic Percussion" with "Chromatic Perc."
        inst_labels = [label.replace("Chromatic Percussion", "Chromatic Perc.") for label in inst_labels]
        ax_inst.set_xticks(range(len(inst_labels)))
        ax_inst.set_xticklabels(inst_labels, rotation=XTICKLABEL_ROTATION, ha='right')
        
        # Pitch distribution (right - unconstrained event) - highlight C#, F, G# in green
        ax_pitch = axes[row_idx, 1]
        pitch_values = [int(token.split(":")[1]) for token in r["pitch_vocab"]]
        # C#=1, F=5, G#=8 in pitch class (mod 12)
        colors = ['green' if (p % 12) in [1, 5, 8] else 'lightsteelblue' for p in pitch_values]
        ax_pitch.bar(range(len(r["pitch_probs"])), r["pitch_probs"], color=colors)
        ax_pitch.set_title("Unconstrained event, pitch probability distribution", fontsize=TITLE_FONT_SIZE, pad=TITLE_PADDING)
        ax_pitch.set_xlabel("Pitch", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_pitch.set_ylabel("Probability", fontsize=AXIS_LABEL_FONT_SIZE)
        ax_pitch.set_ylim(0, 0.5)
        ax_pitch.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
        # Show fewer x-tick labels
        tick_positions = list(range(0, len(r["pitch_vocab"]), 12))
        tick_labels = [r["pitch_vocab"][i].split(":")[1] for i in tick_positions]
        ax_pitch.set_xticks(tick_positions)
        ax_pitch.set_xticklabels(tick_labels)
    
    pitch_plot_base = os.path.join(output_dir, "pitch_instrument_comparison")
    fig.savefig(f"{pitch_plot_base}.png", dpi=150, bbox_inches='tight')
    fig.savefig(f"{pitch_plot_base}.pdf", bbox_inches='tight')
    plt.show()
    
    del model
    torch.cuda.empty_cache()


def run_drum_pitch_onset_experiment(model_type):
    """
    Experiment: 32 drum events, constrain one event's pitch to kick+crash or snare+clap,
    observe the onset time distribution for that event.
    """
    print(f"\n{'='*60}")
    print(f"Drum Pitch -> Onset Experiment: {model_type}")
    print(f"{'='*60}")
    
    model = load_model(
        model_type=model_type,
        epochs=150,
        device=DEVICE
    )
    
    if USE_FP16:
        model = model.half()
    
    N_EVENTS = model.tokenizer.config["max_notes"]
    N_ACTIVE_EVENTS = 32
    ec = lambda: MusicalEventConstraint(model.tokenizer)
    
    output_dir = f"neighbour_constraints/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for condition_name, condition_info in DRUM_PITCH_CONDITIONS.items():
        pitches = condition_info["pitches"]
        label = condition_info["label"]
        constraint_text = condition_info.get("constraint_text", f"pitch ∈ {{{', '.join(sorted(pitches))}}}")
        print(f"  Running: {label}")
        
        # Build constraints: 32 events, all drums, first event has pitch constraint
        e = []
        # Event 0: Constrained to specific drum pitches
        e += [ec()
              .intersect({"instrument": {"Drums"}, "pitch": pitches})
              .force_active()]
        # Events 1-31: Drums, no pitch constraint
        e += [ec()
              .intersect({"instrument": {"Drums"}})
              .force_active() 
              for _ in range(N_ACTIVE_EVENTS - 1)]
        # Pad remaining as inactive
        e += [ec().force_inactive() for _ in range(N_EVENTS - len(e))]
        
        mask = model.tokenizer.event_constraints_to_mask(e).to(DEVICE)
        out_logits = model.forward(mask.float())
        
        # Get constrained event 0 probabilities
        event_0_logits = out_logits[0, 0]
        event_0_probs = torch.softmax(event_0_logits, dim=-1)
        
        # Get onset/global_tick distribution
        onset_vocab = [token for token in model.tokenizer.vocab if "onset/global_tick:" in token and token != "onset/global_tick:-"]
        onset_index = model.tokenizer.note_attribute_order.index("onset/global_tick")
        onset_token_indices = [model.tokenizer.vocab.index(token) for token in onset_vocab]
        onset_probs = event_0_probs[onset_index, onset_token_indices].cpu().detach().numpy()
        onset_probs = onset_probs / onset_probs.sum()
        
        results[condition_name] = {
            "label": label,
            "onset_vocab": onset_vocab,
            "onset_probs": onset_probs,
        }
    
    # Create figure: 3 rows (conditions) x 1 col (onset/global_tick)
    fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE)
    plt.subplots_adjust(hspace=SUBPLOT_ROW_SPACING_DRUMS)
    
    ticks_per_beat = model.tokenizer.config.get("ticks_per_beat", 24)
    
    constraint_prefixes = ["a)", "b)", "c)"]
    for row_idx, condition_name in enumerate(["kick_crash", "snare_clap", "toms"]):
        r = results[condition_name]
        ax = axes[row_idx]
        
        # Add constraint text above subplot
        prefix = constraint_prefixes[row_idx] if row_idx < len(constraint_prefixes) else ""
        constraint_text = f"{prefix} Constraint: {DRUM_PITCH_CONDITIONS[condition_name]['constraint_text']}".strip()
        ax.text(
            0.5,
            CONSTRAINT_LABEL_Y_POSITION,
            constraint_text,
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=CONSTRAINT_LABEL_FONT_SIZE,
            fontweight='bold',
            color='#2c3e50',
            clip_on=False,
        )
        
        onset_labels = [v.split(":")[1] for v in r["onset_vocab"]]
        ax.bar(range(len(r["onset_probs"])), r["onset_probs"], color='steelblue')
        ax.set_title("Constrained event, onset/global_tick probability distribution", fontsize=TITLE_FONT_SIZE, pad=TITLE_PADDING)
        ax.set_xlabel("Onset Tick", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("Probability", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylim(0, 0.1)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
        # Show fewer ticks (every beat)
        n_ticks = len(onset_labels)
        tick_positions = list(range(0, n_ticks, ticks_per_beat))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([onset_labels[i] for i in tick_positions])
    
    onset_plot_base = os.path.join(output_dir, "drum_pitch_onset_comparison")
    fig.savefig(f"{onset_plot_base}.png", dpi=150, bbox_inches='tight')
    fig.savefig(f"{onset_plot_base}.pdf", bbox_inches='tight')
    plt.show()
    
    del model
    torch.cuda.empty_cache()


#%%
if __name__ == "__main__":
    os.makedirs("neighbour_constraints", exist_ok=True)
    
    for model_type in MODEL_TYPES:
        run_experiments_for_model(model_type)
        run_drum_pitch_onset_experiment(model_type)
    
    print(f"\nDone. Check neighbour_constraints/<model_type>/")

# %%
