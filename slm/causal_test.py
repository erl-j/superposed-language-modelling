#%%
#from h_causal import HierarchicalCausalDecoderModel
from h_causal_w_prior import HierarchicalCausalDecoderModel

device = "cuda:0"
# Load the model
model = HierarchicalCausalDecoderModel.load_from_checkpoint(
    # "../checkpoints/glowing-lion-18/last.ckpt",
    # "../checkpoints/copper-cosmos-4/last.ckpt",
    #"../checkpoints/celestial-microwave-7/last.ckpt",
    "../checkpoints/still-microwave-5/last.ckpt",
    map_location=device,
)

mask = model.tokenizer.constraint_mask(
    tags = ["pop"],
    tempos = ["128"],
    instruments =["Drums"],
    # scale = "G major",
    min_notes=10,
    max_notes=290,
    min_notes_per_instrument=50
)[None,:]

# mask = model.tokenizer.get_format_mask()[None,...]
x = model.sample(mask,temperature=0.95)

from util import preview_sm
x_sm = model.tokenizer.decode(x.flatten())
preview_sm(x_sm)
print(x_sm.note_num())
# %%
print(x.shape)
print(model.tokenizer.indices_to_tokens(x.flatten()))
# %%
