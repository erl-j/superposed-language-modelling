import os
import sys

checkpoints = [
"checkpoints/different-firefly-705/every25/epoch=49",
"checkpoints/different-firefly-705/every25/epoch=99",
"checkpoints/different-firefly-705/every25/epoch=149",
"checkpoints/grateful-moon-708/every25/epoch=49",
"checkpoints/skilled-frost-711/every25/epoch=99",
"checkpoints/skilled-frost-711/every25/epoch=149",
"checkpoints/rosy-serenity-707/every25/epoch=49",
"checkpoints/logical-butterfly-710/every25/epoch=99",
"checkpoints/logical-butterfly-710/every25/epoch=149",
"checkpoints/denim-microwave-714/every25/epoch=49",
"checkpoints/denim-microwave-714/every25/epoch=99",
"checkpoints/denim-microwave-714/every25/epoch=149",
"checkpoints/usual-fire-530/last.ckpt",
"checkpoints/smart-brook-552/last.ckpt",
"checkpoints/toasty-bush-529/last.ckpt",
"checkpoints/crisp-paper-617/last.ckpt",
"checkpoints/balmy-deluge-532/last.ckpt",
"checkpoints/pretty-armadillo-542/last.ckpt",
"checkpoints/colorful-sun-548/last.ckpt",
"checkpoints/smooth-meadow-615/last.ckpt",
"checkpoints/golden-monkey-568/last.ckpt",
"checkpoints/driven-planet-576/last.ckpt",
"checkpoints/fancy-paper-577/last.ckpt",
"checkpoints/woven-pyramid-588/last.ckpt",
"checkpoints/glowing-pine-623/last.ckpt",
"checkpoints/eternal-dawn-640/last.ckpt",
"checkpoints/stellar-sun-634/last.ckpt",
"checkpoints/fresh-glitter-641/last.ckpt",
"checkpoints/warm-spaceship-643/last.ckpt",
"checkpoints/leafy-galaxy-677/last.ckpt",
"checkpoints/lilac-paper-676/last.ckpt"
"checkpoints/different-firefly-705/every25/epoch=49-step=189050-val/accuracy@1=0.94235.ckpt",
"checkpoints/different-firefly-705/every25/epoch=149-step=567150-val/accuracy@1=0.94588.ckpt"
"checkpoints/noble-sea-291/epoch=65-step=95238-val/loss_epoch=0.36076.ckpt"
"checkpoints/trim-water-280/epoch=132-step=191919-val/loss_epoch=0.14.ckpt"
"checkpoints/noble-plasma-309/epoch=144-step=209235-val/loss_epoch=0.13589.ckpt"
"checkpoints/avid-fog-315/epoch=40-step=29602-val/loss_epoch=0.16085.ckpt"
"checkpoints/frosty-galaxy-297/epoch=29-step=43290-val/loss_epoch=0.15267.ckpt"
"checkpoints/denim-paper-306/epoch=39-step=72150-val/loss_epoch=0.36424.ckpt"
"checkpoints/fast-paper-307/epoch=32-step=51948-val/loss_epoch=0.35981.ckpt"
"checkpoints/honest-snow-308/epoch=32-step=95238-val/loss_epoch=0.35076.ckpt"
"checkpoints/distinctive-moon-314/epoch=162-step=235209-val/loss_epoch=0.20269.ckpt"
]

runs_to_keep = list(set(["/".join(c.split("/")[:2]) for c in checkpoints]))

# get combined size of all checkpoints
total_size = 0
print(runs_to_keep)

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

total_size = 0
for c in runs_to_keep:
    print(c)
    # get size of directory in bytes
    size = get_size(c)
    total_size += size
    print(f"Size of {c}: {size / 1024 / 1024 / 1024:.2f} GB")
    # copy dir to gold_checkpoints
    os.system(f"cp -r {c} gold_checkpoints")
    print(f"cp -r {c} gold_checkpoints")

    

# print total size in human readable format
print(f"Total size of checkpoints to keep: {total_size / 1024 / 1024 / 1024:.2f} GB")
print(total_size)
