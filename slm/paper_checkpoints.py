import os
import sys

CHECKPOINTS = {
    # "slm": "../checkpoints/usual-fire-530/last.ckpt",
    "slm": "checkpoints/smart-brook-552/last.ckpt",
    "mlm": "checkpoints/toasty-bush-529/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "checkpoints/balmy-deluge-532/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "checkpoints/pretty-armadillo-542/last.ckpt",
    "slm_wo_enforce_constraint_in_fwd": "checkpoints/colorful-sun-548/last.ckpt",
    "slm_not_norm_first": "checkpoints/rural-oath-549/last.ckpt",
}
# if checkpoints is in parent directory, change checkpoints to ../checkpoints
for k, v in CHECKPOINTS.items():
    if not os.path.exists(v):
        CHECKPOINTS[k] = "../" + v
