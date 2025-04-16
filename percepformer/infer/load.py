import torch
from pathlib import Path
from ..model.model import create_model


def load_checkpoint(checkpoint_path, map_location=None):
    """
    Load a PyTorch checkpoint (.pth file) from the given path.

    Args:
        checkpoint_path (str): Path to the .pth file.
        map_location (str or torch.device, optional): Device to map the checkpoint tensors to.
            e.g., 'cpu' or 'cuda'. Default is None, which uses the default device.

    Returns:
        dict: Loaded checkpoint (usually a state_dict or a dictionary with model and optimizer states).
    """

    root_dir = Path(__file__).resolve().parents[2]
    if not checkpoint_path.is_absolute():
        checkpoint_path = root_dir / checkpoint_path

    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        print(f"Checkpoint loaded successfully from: {checkpoint_path}")
        return checkpoint
    except FileNotFoundError:
        print(f"Checkpoint file not found: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")


def load_model_from_checkpoint(checkpoint):
    """
    Load a model from a loaded PyTorch checkpoint dictionary.

    Args:
        checkpoint (dict): Loaded checkpoint containing 'config' and 'model_state_dict'.

    Returns:
        model (torch.nn.Module): Model with weights loaded from the checkpoint.
    """
    # Extract configuration dictionary
    config = checkpoint["config"]

    # Create model using the parameters in the config
    model = create_model(**config["model"]["parameters"])

    # Load the pretrained weights into the model
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Model loaded successfully from checkpoint.")
    return model
