
good_checkpoints = {
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
"slm_full_150epochs": "gold_checkpoints/denim-microwave-714/every25/epoch=149"
}

import os
import shutil
import glob

# Create good_checkpoints directory
os.makedirs("good_checkpoints", exist_ok=True)

# Copy each checkpoint
for name, base_path in good_checkpoints.items():
    # Find the actual checkpoint file in the directory
    checkpoint_dir = f"{base_path}-step=*"
    matching_dirs = glob.glob(checkpoint_dir)
    
    if matching_dirs:
        checkpoint_dir = matching_dirs[0]  # Take the first match
        
        # Find the .ckpt file in the checkpoint directory
        ckpt_files = glob.glob(f"{checkpoint_dir}/*.ckpt")
        
        if ckpt_files:
            ckpt_file = ckpt_files[0]  # Take the first .ckpt file
            dest_path = f"good_checkpoints/{name}.ckpt"
            shutil.copy2(ckpt_file, dest_path)
            print(f"Copied {ckpt_file} to {dest_path}")
        else:
            print(f"No .ckpt file found in {checkpoint_dir}")
    else:
        print(f"No checkpoint found for {name} at {base_path}")
        