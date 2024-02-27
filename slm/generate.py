#%%
from train import DecoderOnlyModel

device = "cuda:6"

# Load the model
model = DecoderOnlyModel.load_from_checkpoint("../checkpoints/glamorous-water-14/last.ckpt"
)

# Move the model to the device
model = model.to(device)

# Generate a sequence
a = model.tokenizer.get_format_mask()[None,...].to(model.device)

#%%


# Generate a sequence
sequence = model.generate(a, max_len=model.tokenizer.total_len, temperature=1.0)
# %%

token_idx = sequence[0].cpu().numpy()


# argmax
token_idx = token_idx.argmax(axis=1)


# decode
sm = model.tokenizer.decode(token_idx)


# save the sequence
sm.dump_midi("../artefacts/generated.mid")
# %%
