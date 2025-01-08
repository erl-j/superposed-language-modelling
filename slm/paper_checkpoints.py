import os
import sys

CHECKPOINTS = {
    # "slm": "../checkpoints/usual-fire-530/last.ckpt",
    # "slm": "checkpoints/smart-brook-552/last.ckpt",
    # "mlm": "checkpoints/toasty-bush-529/last.ckpt",
    "mlm_not_norm_first": "checkpoints/crisp-paper-617/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "checkpoints/balmy-deluge-532/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "checkpoints/pretty-armadillo-542/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "checkpoints/colorful-sun-548/last.ckpt",
    "slm_not_norm_first": "checkpoints/smooth-meadow-615/last.ckpt",
    # "slm_tiny_not_norm_first": "./checkpoints/golden-monkey-568/last.ckpt",
    # "slm_tiny_x**1/4" : "checkpoints/driven-planet-576/last.ckpt",
    # "slm_tiny_x**1/2" : "checkpoints/fancy-paper-577/last.ckpt",
    # "slm_w_proj" : "checkpoints/woven-pyramid-588/last.ckpt",
    "slm_w_mixed_superposition": "checkpoints/glowing-pine-623/last.ckpt",
}
# if checkpoints is in parent directory, change checkpoints to ../checkpoints
for k, v in CHECKPOINTS.items():
    if not os.path.exists(v):
        CHECKPOINTS[k] = "../" + v
