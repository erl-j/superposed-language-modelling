#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model generation jobs in tmux sessions')
    parser.add_argument('--auto-gpu', action='store_true', help='Automatically allocate available GPUs', default=False)
    parser.add_argument('--gpus', type=str, help='Comma-separated list of GPU IDs to use (e.g., "0,1,3,5")')
    parser.add_argument('--models', type=str, help='Comma-separated list of models (optional)', default=None)
    args = parser.parse_args()

    # List of models to process
    default_models = [
        "slm_sparse_150epochs",
        "slm_full_150epochs",
        "slm_mixed_150epochs",
        "mlm_150epochs"
    ]
    
    # Use custom models if provided
    if args.models:
        models = args.models.split(',')
    else:
        models = default_models

    # GPU allocation
    if args.auto_gpu:
        # Get number of available GPUs
        nvidia_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode()
        n_gpus = len(nvidia_output.strip().split('\n'))
        print(f"Found {n_gpus} GPUs")
        gpu_ids = list(range(n_gpus))
    elif args.gpus:
        # Use specified GPUs
        gpu_ids = [gpu.strip() for gpu in args.gpus.split(',')]
        print(f"Using GPUs: {', '.join(gpu_ids)}")
    else:
        # Default GPU mappings
        gpu_mapping = {
            "model1": "3",
            "model2": "4",
            "model3": "5",
            "model4": "6"
        }
        gpu_ids = None

    # Current directory
    current_dir = os.getcwd()

    # Create a tmux session for each model
    for i, model in enumerate(models):
        # Create simple session name
        session_name = f"gen_{model.replace('.', '_')}"
        
        # Determine GPU ID
        if gpu_ids is not None:
            gpu_id = gpu_ids[i % len(gpu_ids)]
        else:
            gpu_id = gpu_mapping.get(model, "0")
        
        # Create tmux session and run command
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name])
        subprocess.run(["tmux", "send-keys", "-t", session_name, f"cd {current_dir}", "C-m"])
        cmd = f"python slm/render_applications_pl.py --render-ground-truth {i==0} --model {model} --gpu {gpu_id}"
        subprocess.run(["tmux", "send-keys", "-t", session_name, cmd, "C-m"])
        
        print(f"Started {model} on GPU {gpu_id} in session {session_name}")

    print("\nAll processes started. Use 'tmux ls' to see sessions")
    print("Attach: tmux attach -t SESSION_NAME | Switch: Ctrl+B s | Detach: Ctrl+B d")

if __name__ == "__main__":
    main()