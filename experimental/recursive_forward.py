# %% Imports
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.load_model_util import load_model
from slm.constraints.core import MusicalEventConstraint
from slm.constraints.templates import breakbeat
from slm.util import top_k_top_p_filtering, sm_fix_overlap_notes, plot_piano_roll

# %% Load Model
device = "cuda:7" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = load_model(model_type="slm_mixed", epochs=150, device=device)
model.eval()
tokenizer = model.tokenizer
vocab_size = len(tokenizer.vocab)

# %% Config
batch_size = 1
num_recursion_steps = 100
temperature = 0.9
topp = 1.0
noise_scale = 1e-8 # Uniform noise scale at step 0
ec = lambda: MusicalEventConstraint(tokenizer)
events = breakbeat(
    e=[], ec=ec, n_events=tokenizer.config["max_notes"],
    beat_range=[0, 16], pitch_range=[0, 128], drums=True,
    tag="pop", tempo=120,
)

mask = tokenizer.event_constraints_to_mask(events).to(device)

entropies = []

print(mask.shape)
current_input = mask
with torch.no_grad():
    for step in range(num_recursion_steps):
        logits = model(current_input)

        # print logits min and max
        print(logits.min(), logits.max())

        # Apply top-p filtering
        original_shape = logits.shape
        flat_logits = logits.reshape(-1, logits.size(-1))  # [batch*seq*dim, vocab]
        filtered_logits = top_k_top_p_filtering(flat_logits, top_k=0, top_p=topp)
        logits = filtered_logits.reshape(original_shape)
        
        next_token_logits = logits / temperature
        
        probs = F.softmax(next_token_logits, dim=-1)

        # add noise
        noise = torch.rand_like(probs) * noise_scale
        probs = probs + noise
        probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True)
        # Renormalize
    
        
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        # Mask out -inf to avoid nan in entropy
        valid_mask = ~torch.isinf(log_probs)
        entropy_terms = torch.where(valid_mask, probs * log_probs, torch.zeros_like(probs))
        entropy = -entropy_terms.sum(dim=-1).mean().item()
        entropies.append(entropy)
        current_input = probs
        print(f"Step {step+1}/{num_recursion_steps}: Entropy = {entropy:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_recursion_steps + 1), entropies, 'b-', linewidth=2)
plt.xlabel('Recursion Step')
plt.ylabel('Average Entropy (nats)')
plt.title('Entropy vs Recursion Step')
plt.grid(True, alpha=0.3)
plt.tight_layout()
entropy_base = 'recursive_entropy'
plt.savefig(f'{entropy_base}.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{entropy_base}.pdf', bbox_inches='tight')
plt.show()

print("\nSampling from final distribution...")
# Sample from final probabilities
final_sample = torch.multinomial(current_input.reshape(-1, current_input.size(-1)), num_samples=1)
final_sample = final_sample.reshape(current_input.shape[:-1])

# Decode to symusic
sm = tokenizer.decode(final_sample[0])
sm = sm_fix_overlap_notes(sm)

# Save MIDI
sm.dump_midi('recursive_output.mid')
print("Saved to recursive_output.mid")

# Plot piano roll
fig = plot_piano_roll(sm)
roll_base = 'recursive_output_piano_roll'
plt.savefig(f'{roll_base}.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{roll_base}.pdf', bbox_inches='tight')
print("Saved piano roll to recursive_output_piano_roll.png and .pdf")
plt.show()

# %%
