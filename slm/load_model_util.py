
def load_model(
    model_type="slm_mixed",
    epochs=150,
    device="cuda",
    cache_dir=None
):
    """
    Load a model from a checkpoint.
    
    Args:
        model_type (str): Type of model to load. Options: "mlm", "slm_sparse", "slm_mixed".
        epochs (int): Number of epochs. Options: 50, 100, 150.
        device (str): Device to load the model on.
        cache_dir (str, optional): Directory to cache checkpoints.
        
    Returns:
        SuperposedLanguageModel: The loaded model.
    """
    import os
    import requests
    from pathlib import Path
    from .train import TrainingWrapper

    if model_type not in ["mlm", "slm_sparse", "slm_mixed"]:
        raise ValueError(f"Invalid model_type: {model_type}. Options: 'mlm', 'slm_sparse', 'slm_mixed'")
    
    if epochs not in [50, 100, 150]:
        raise ValueError(f"Invalid epochs: {epochs}. Options: 50, 100, 150")
        
    # Construct checkpoint filename
    ckpt_name = f"{model_type}_{epochs}epochs.ckpt"
    
    # Determine cache directory
    if cache_dir is None:
        # Default to ~/.cache/slm
        cache_dir = Path.home() / ".cache" / "slm"
    else:
        cache_dir = Path(cache_dir)
        
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / ckpt_name
    
    # Download if not exists
    if not ckpt_path.exists():
        url = f"https://huggingface.co/erl-j/slm/resolve/main/{ckpt_name}"
        print(f"Downloading checkpoint from {url} to {ckpt_path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(ckpt_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download checkpoint: {e}")
            if ckpt_path.exists():
                ckpt_path.unlink()
            raise
            
    # Load model
    print(f"Loading model from {ckpt_path}...")
    wrapper = TrainingWrapper.load_from_checkpoint(
        ckpt_path,
        map_location=device,
    )
    
    return wrapper.model

