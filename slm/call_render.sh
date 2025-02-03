#!/bin/bash

# List of models to process - you'll need to replace these with your actual model names
models=(
    "slm_sparse_100epochs"
    "slm_sparse_150epochs"
    "slm_full_100epochs"
    "slm_full_150epochs"
    "slm_mixed_100epochs"
    "slm_mixed_150epochs"
    "mlm_100epochs"
    "mlm_150epochs"
)

# Check if running with GPU allocation
if [ "$1" == "--auto-gpu" ]; then
    # Get number of available GPUs
    n_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo "Found $n_gpus GPUs"
else
    # Use explicit GPU mappings
    declare -A gpu_mapping=(
        ["model1"]="0"
        ["model2"]="1"
        ["model3"]="2"
        ["model4"]="3"
    )
fi

# Create a new tmux session for each model
for model in "${models[@]}"; do
    # Generate a clean session name from model name (remove any problematic characters)
    session_name="gen_${model//[^a-zA-Z0-9]/_}"
    
    # If using auto GPU allocation, calculate GPU ID
    if [ "$1" == "--auto-gpu" ]; then
        gpu_id=$((i % n_gpus))
    else
        gpu_id=${gpu_mapping[$model]}
    fi
    
    # Create new tmux session in detached state
    tmux new-session -d -s "$session_name"
    
    # Navigate to correct directory and activate environment if needed
    tmux send-keys -t "$session_name" "cd $(pwd)" C-m
    
    # Optional: Activate virtual environment if needed
    # tmux send-keys -t "$session_name" "conda activate your_env" C-m
    
    # Run the generation script
    tmux send-keys -t "$session_name" "python slm/render_applications_pl.py --model $model --gpu $gpu_id" C-m 
    
    echo "Started generation for $model on GPU $gpu_id in tmux session $session_name"
done

echo "All generation processes started. Use 'tmux ls' to see sessions"
echo "To attach to a session: tmux attach-session -t session_name"
echo "To switch between sessions: Ctrl+B s"
echo "To detach from a session: Ctrl+B d"