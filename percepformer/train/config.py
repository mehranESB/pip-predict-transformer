import os

current_dir = os.path.dirname(os.path.abspath(__file__))

config = {
    # Dataset configurations
    "data": {
        # Paths to .pkl datasets
        "pkl_pathes": [
            os.path.join(current_dir, "../../DATA/pip/EURUSD-1h.pkl"),
            os.path.join(current_dir, "../../DATA/pip/EURUSD-15m.pkl"),
        ],
        # Configuration for data augmentation transformations
        "transformation": {
            "mirror_reflect": {"mode": "random"},  # Randomly apply mirror reflection
            "add_uniform_score": {
                "mu": 0.0235822,  # Location parameter for GEV distribution
                "sigma": 0.0227505,  # Scale parameter for GEV distribution
                "xi": 0.652834,  # Shape parameter for GEV distribution
            },
        },
        # Split ratios for dataset
        "train_ratio": 0.8,  # Proportion of training data
        "valid_ratio": 0.15,  # Proportion of validation data
    },
    # Model configuration
    "model": {
        # Parameters to create the model instance
        "parameters": {
            "in_channels": 5,
            "d_model": 32,
            "num_blocks": 2,
            "num_layers": 2,
            "num_groups": 4,
            "embed_act_fun": "tanh",
            "act_fun": "relu",
            "nhead": 4,
            "num_encoder_layers": 4,
            "dim_feedforward": 128,
            "dropout": 0.1,
        },
        # Whether to load weights from a pretrained checkpoint
        "load_from_checkpoint": False,  # Default: False
        # Path to the checkpoint file (if loading pretrained weights)
        "checkpoint_path": None,  # Default: None
        # Flag to indicate if the model should learn to sort iteration
        "learn_to_sort": False,  # Default: False
    },
}
