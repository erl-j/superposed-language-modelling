import os
import sys

CHECKPOINTS = {
    "slm_sparse_50epochs": "gold_checkpoints/different-firefly-705/every25/epoch=49",
    "slm_sparse_100epochs": "gold_checkpoints/different-firefly-705/every25/epoch=99",
    "slm_sparse_150epochs": "gold_checkpoints/different-firefly-705/every25/epoch=149",
    "mlm_50epochs": "gold_checkpoints/grateful-moon-708/every25/epoch=49",
    "mlm_100epochs": "gold_checkpoints/skilled-frost-711/every25/epoch=99",
    "mlm_150epochs": "gold_checkpoints/skilled-frost-711/every25/epoch=149",
    "slm_mixed_50epochs": "gold_checkpoints/rosy-serenity-707/every25/epoch=49",
    "slm_mixed_100epochs": "gold_checkpoints/logical-butterfly-710/every25/epoch=99",
    "slm_mixed_150epochs": "gold_checkpoints/logical-butterfly-710/every25/epoch=149",
    "slm_full_50epochs": "gold_checkpoints/denim-microwave-714/every25/epoch=49",
    "slm_full_100epochs": "gold_checkpoints/denim-microwave-714/every25/epoch=99",
    "slm_full_150epochs": "gold_checkpoints/denim-microwave-714/every25/epoch=149",
    # "slm": "../gold_checkpoints/usual-fire-530/last.ckpt",
    # "slm": "gold_checkpoints/smart-brook-552/last.ckpt",
    # "mlm": "gold_checkpoints/toasty-bush-529/last.ckpt",
    # "mlm": "gold_checkpoints/crisp-paper-617/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "gold_checkpoints/balmy-deluge-532/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "gold_checkpoints/pretty-armadillo-542/last.ckpt",
    # "slm_wo_enforce_constraint_in_fwd": "gold_checkpoints/colorful-sun-548/last.ckpt",
    # "slm": "gold_checkpoints/smooth-meadow-615/last.ckpt",
    # "slm_tiny_not_norm_first": "./gold_checkpoints/golden-monkey-568/last.ckpt",
    # "slm_tiny_x**1/4" : "gold_checkpoints/driven-planet-576/last.ckpt",
    # "slm_tiny_x**1/2" : "gold_checkpoints/fancy-paper-577/last.ckpt",
    # "slm_w_proj" : "gold_checkpoints/woven-pyramid-588/last.ckpt",
    # "slm_w_mixed_superposition": "gold_checkpoints/glowing-pine-623/last.ckpt",
    # "slm_w_mixed_superposition_2": "gold_checkpoints/eternal-dawn-640/last.ckpt",
    # "mlm_w_mixed_superposition": "gold_checkpoints/stellar-sun-634/last.ckpt",
    # "slm_simulated_mlm" :  "gold_checkpoints/fresh-glitter-641/last.ckpt",
    # "slm_simulated_mixed_mlm": "gold_checkpoints/warm-spaceship-643/last.ckpt",
    # "slm_mixed_ratio": "./gold_checkpoints/leafy-galaxy-677/last.ckpt",
    # "slm_mixed_ratio_w_shared": "gold_checkpoints/lilac-paper-676/last.ckpt"
    # "slm_sparse_sup_50":"gold_checkpoints/different-firefly-705/every25/epoch=49-step=189050-val/accuracy@1=0.94235.ckpt",
    # "slm_sparse_sup_150": "gold_checkpoints/different-firefly-705/every25/epoch=149-step=567150-val/accuracy@1=0.94588.ckpt"
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
