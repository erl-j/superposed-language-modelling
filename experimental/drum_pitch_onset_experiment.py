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

# Drum pitch conditions (GM drums)
# Kick: 36, Snare: 38, Clap: 39, Crash: 49, 57
PITCH_CONDITIONS = {
    "kick_crash": {
        "pitches": {"36 (Drums)", "49 (Drums)", "57 (Drums)"},
        "label": "Kick + Crash (36, 49, 57)"
    },
    "snare_clap": {
        "pitches": {"38 (Drums)", "39 (Drums)"},
        "label": "Snare + Clap (38, 39)"
    },
}

N_ACTIVE_EVENTS = 32

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
    
    output_dir = f"drum_pitch_onset/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for condition_name, condition_info in PITCH_CONDITIONS.items():
        pitches = condition_info["pitches"]
        label = condition_info["label"]
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
        
        # Get onset/beat distribution
        onset_beat_vocab = [token for token in model.tokenizer.vocab if "onset/beat:" in token and token != "onset/beat:-"]
        onset_beat_index = model.tokenizer.note_attribute_order.index("onset/beat")
        onset_beat_token_indices = [model.tokenizer.vocab.index(token) for token in onset_beat_vocab]
        onset_beat_probs = event_0_probs[onset_beat_index, onset_beat_token_indices].cpu().detach().numpy()
        onset_beat_probs = onset_beat_probs / onset_beat_probs.sum()
        
        # Get onset/tick distribution  
        onset_tick_vocab = [token for token in model.tokenizer.vocab if "onset/tick:" in token and token != "onset/tick:-"]
        onset_tick_index = model.tokenizer.note_attribute_order.index("onset/tick")
        onset_tick_token_indices = [model.tokenizer.vocab.index(token) for token in onset_tick_vocab]
        onset_tick_probs = event_0_probs[onset_tick_index, onset_tick_token_indices].cpu().detach().numpy()
        onset_tick_probs = onset_tick_probs / onset_tick_probs.sum()
        
        results[condition_name] = {
            "label": label,
            "onset_beat_vocab": onset_beat_vocab,
            "onset_beat_probs": onset_beat_probs,
            "onset_tick_vocab": onset_tick_vocab, 
            "onset_tick_probs": onset_tick_probs,
        }
    
    # Create figure: 2 rows (conditions) x 2 cols (beat, tick)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Onset Time Distribution by Drum Pitch Constraint\n{model_type}", fontsize=18, fontweight='bold')
    
    for row_idx, condition_name in enumerate(["kick_crash", "snare_clap"]):
        r = results[condition_name]
        
        # Onset/beat distribution (left column)
        ax_beat = axes[row_idx, 0]
        beat_labels = [v.split(":")[1] for v in r["onset_beat_vocab"]]
        ax_beat.bar(range(len(r["onset_beat_probs"])), r["onset_beat_probs"], color='steelblue')
        ax_beat.set_title(f"{r['label']}\nonset/beat distribution", fontsize=14)
        ax_beat.set_xlabel("Beat", fontsize=12)
        ax_beat.set_ylabel("Probability", fontsize=12)
        ax_beat.set_ylim(0, max(r["onset_beat_probs"]) * 1.2)
        # Show fewer ticks
        n_beats = len(beat_labels)
        tick_step = max(1, n_beats // 16)
        tick_positions = list(range(0, n_beats, tick_step))
        ax_beat.set_xticks(tick_positions)
        ax_beat.set_xticklabels([beat_labels[i] for i in tick_positions])
        
        # Onset/tick distribution (right column)
        ax_tick = axes[row_idx, 1]
        tick_labels = [v.split(":")[1] for v in r["onset_tick_vocab"]]
        ax_tick.bar(range(len(r["onset_tick_probs"])), r["onset_tick_probs"], color='coral')
        ax_tick.set_title(f"{r['label']}\nonset/tick distribution", fontsize=14)
        ax_tick.set_xlabel("Tick", fontsize=12)
        ax_tick.set_ylabel("Probability", fontsize=12)
        ax_tick.set_ylim(0, max(r["onset_tick_probs"]) * 1.2)
        ax_tick.set_xticks(range(len(tick_labels)))
        ax_tick.set_xticklabels(tick_labels)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/drum_pitch_onset_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    del model
    torch.cuda.empty_cache()

#%%
if __name__ == "__main__":
    os.makedirs("drum_pitch_onset", exist_ok=True)
    
    for model_type in MODEL_TYPES:
        run_experiments_for_model(model_type)
    
    print(f"\nDone. Check drum_pitch_onset/<model_type>/drum_pitch_onset_comparison.png")

# %%

