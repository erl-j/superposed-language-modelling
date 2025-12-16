#%%
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from slm import load_model
from slm.constraints.core import MusicalEventConstraint

USE_FP16 = False
DEVICE = "cuda:7"

MODEL_TYPES = ["slm_full", "slm_mixed", "slm_sparse"]

# C major triad (C, E, G) at different octaves
# MIDI: C=0,12,24,36,48,60,72,84,96... E=C+4, G=C+7
OCTAVE_CONDITIONS = {
    "low": {"pitches": [36, 40, 43], "label": "C2-E2-G2 (Low)"},      # C2, E2, G2
    "mid": {"pitches": [60, 64, 67], "label": "C4-E4-G4 (Mid)"},      # C4, E4, G4
    "high": {"pitches": [84, 88, 91], "label": "C6-E6-G6 (High)"},    # C6, E6, G6
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
        e += [ec().force_active()]
        # Pad remaining as inactive
        e += [ec().force_inactive() for _ in range(N_EVENTS - len(e))]
        
        mask = model.tokenizer.event_constraints_to_mask(e).to(DEVICE)
        out_logits = model.forward(mask.float())
        
        # Get event B (unconstrained) probabilities
        event_b_logits = out_logits[0, 1]
        event_b_probs = torch.softmax(event_b_logits, dim=-1)
        
        # Pitch distribution
        pitch_vocab = [token for token in model.tokenizer.vocab 
                       if "pitch:" in token and "-" not in token and "Drums" not in token]
        pitch_index = model.tokenizer.note_attribute_order.index("pitch")
        pitch_token_indices = [model.tokenizer.vocab.index(token) for token in pitch_vocab]
        pitch_probs = event_b_probs[pitch_index, pitch_token_indices].cpu().detach().numpy()
        
        # Instrument distribution
        instrument_vocab = [token for token in model.tokenizer.vocab if "instrument:" in token]
        instrument_index = model.tokenizer.note_attribute_order.index("instrument")
        instrument_token_indices = [model.tokenizer.vocab.index(token) for token in instrument_vocab]
        instrument_probs = event_b_probs[instrument_index, instrument_token_indices].cpu().detach().numpy()
        
        results[octave_name] = {
            "label": label,
            "pitch_vocab": pitch_vocab,
            "pitch_probs": pitch_probs,
            "instrument_vocab": instrument_vocab,
            "instrument_probs": instrument_probs,
        }
    
    # Create figure with 3 rows (octaves) x 2 cols (pitch, instrument)
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle(f"{model_type} - Event B distributions given C-E-G constraint on Event A", fontsize=14)
    
    for row_idx, octave_name in enumerate(["low", "mid", "high"]):
        r = results[octave_name]
        
        # Pitch distribution - highlight C, E, G in green
        ax_pitch = axes[row_idx, 0]
        pitch_values = [int(token.split(":")[1]) for token in r["pitch_vocab"]]
        # C=0, E=4, G=7 in pitch class (mod 12)
        colors = ['green' if (p % 12) in [0, 4, 7] else 'steelblue' for p in pitch_values]
        ax_pitch.bar(range(len(r["pitch_probs"])), r["pitch_probs"], color=colors)
        ax_pitch.set_title(f"Pitch Distribution - {r['label']}")
        ax_pitch.set_xlabel("Pitch")
        ax_pitch.set_ylabel("Probability")
        ax_pitch.set_ylim(0, 0.5)
        # Show fewer x-tick labels
        tick_positions = list(range(0, len(r["pitch_vocab"]), 12))
        tick_labels = [r["pitch_vocab"][i].split(":")[1] for i in tick_positions]
        ax_pitch.set_xticks(tick_positions)
        ax_pitch.set_xticklabels(tick_labels)
        
        # Instrument distribution
        ax_inst = axes[row_idx, 1]
        inst_labels = [v.split(":")[1] for v in r["instrument_vocab"]]
        ax_inst.bar(range(len(r["instrument_probs"])), r["instrument_probs"], color='coral')
        ax_inst.set_title(f"Instrument Distribution - {r['label']}")
        ax_inst.set_xlabel("Instrument")
        ax_inst.set_ylabel("Probability")
        ax_inst.set_ylim(0, 0.5)
        ax_inst.set_xticks(range(len(inst_labels)))
        ax_inst.set_xticklabels(inst_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "ceg_octave_experiment.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    del model
    torch.cuda.empty_cache()

#%%
if __name__ == "__main__":
    os.makedirs("neighbour_constraints", exist_ok=True)
    
    for model_type in MODEL_TYPES:
        run_experiments_for_model(model_type)
    
    print(f"\nDone. Check neighbour_constraints/<model_type>/ceg_octave_experiment.png")

# %%
