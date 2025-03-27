import os
import sys

CHECKPOINTS = {
    "slm_sparse_50epochs": "checkpoints/different-firefly-705/every25/epoch=49",
    "slm_sparse_100epochs": "checkpoints/different-firefly-705/every25/epoch=99",
    "slm_sparse_150epochs": "checkpoints/different-firefly-705/every25/epoch=149",
    "mlm_50epochs": "checkpoints/grateful-moon-708/every25/epoch=49",
    "mlm_100epochs": "checkpoints/skilled-frost-711/every25/epoch=99",
    "mlm_150epochs": "checkpoints/skilled-frost-711/every25/epoch=149",
    "slm_mixed_50epochs": "checkpoints/rosy-serenity-707/every25/epoch=49",
    "slm_mixed_100epochs": "checkpoints/logical-butterfly-710/every25/epoch=99",
    "slm_mixed_150epochs": "checkpoints/logical-butterfly-710/every25/epoch=149",
    "slm_full_50epochs": "checkpoints/denim-microwave-714/every25/epoch=49",
    "slm_full_100epochs": "checkpoints/denim-microwave-714/every25/epoch=99",
    "slm_full_150epochs": "checkpoints/denim-microwave-714/every25/epoch=149",
    # "slm": "../checkpoints/usual-fire-530/last.ckpt",
    # "slm": "checkpoints/smart-brook-552/last.ckpt",
    # "mlm": "checkpoints/toasty-bush-529/last.ckpt",
    # "mlm": "checkpoints/crisp-paper-617/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "checkpoints/balmy-deluge-532/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "checkpoints/pretty-armadillo-542/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "checkpoints/colorful-sun-548/last.ckpt",
    # "slm": "checkpoints/smooth-meadow-615/last.ckpt",
    # "slm_tiny_not_norm_first": "./checkpoints/golden-monkey-568/last.ckpt",
    # "slm_tiny_x**1/4" : "checkpoints/driven-planet-576/last.ckpt",
    # "slm_tiny_x**1/2" : "checkpoints/fancy-paper-577/last.ckpt",
    # "slm_w_proj" : "checkpoints/woven-pyramid-588/last.ckpt",
    # "slm_w_mixed_superposition": "checkpoints/glowing-pine-623/last.ckpt",
    # "slm_w_mixed_superposition_2": "checkpoints/eternal-dawn-640/last.ckpt",
    # "mlm_w_mixed_superposition": "checkpoints/stellar-sun-634/last.ckpt",
    # "slm_simulated_mlm" :  "checkpoints/fresh-glitter-641/last.ckpt",
    # "slm_simulated_mixed_mlm": "checkpoints/warm-spaceship-643/last.ckpt",
    # "slm_mixed_ratio": "./checkpoints/leafy-galaxy-677/last.ckpt",
    # "slm_mixed_ratio_w_shared": "checkpoints/lilac-paper-676/last.ckpt"
    # "slm_sparse_sup_50":"checkpoints/different-firefly-705/every25/epoch=49-step=189050-val/accuracy@1=0.94235.ckpt",
    # "slm_sparse_sup_150": "checkpoints/different-firefly-705/every25/epoch=149-step=567150-val/accuracy@1=0.94588.ckpt"

}



import glob
# if checkpoints is in parent directory, change checkpoints to ../checkpoints
for k, v in CHECKPOINTS.items():
    paths = glob.glob(v + "**/*.ckpt", recursive=True) + glob.glob( "../" + v + "**/*.ckpt", recursive=True)
    print(paths)
    if len(paths) == 0:
        raise FileNotFoundError(f"Checkpoint not found: {v}")
    else:
        CHECKPOINTS[k] = paths[0]
