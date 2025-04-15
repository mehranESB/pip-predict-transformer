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
    # Optimizer configuration
    "optimizer": {
        # Type of optimizer: Adam, SGD, etc.
        "type": "Adam",  # Default optimizer type
        # Parameters for the optimizer
        "parameters": {
            "lr": 0.001,  # Learning rate
            "betas": (0.9, 0.999),  # Beta parameters for Adam
            "eps": 1e-08,  # Epsilon value
            "weight_decay": 0.0,  # Weight decay (L2 regularization)
            "amsgrad": False,  # Use AMSGrad variant of Adam
        },
        # Load optimizer from checkpoint
        "load_from_checkpoint": False,
        "checkpoint_path": None,  # Path to optimizer checkpoint
    },
    # Training configuration
    "train": {
        "device": "cuda",  # Device to train on: 'cuda' or 'cpu'
        "batch_size": 64,  # Batch size for DataLoaders
        "shuffle": True,  # Shuffle the training data
        "scheduler": {
            "type": "StepLR",  # Scheduler type
            "parameters": {"step_size": 30, "gamma": 0.1},
        },
        "seed": 42,  # random seed for reproducibility
        "epochs": 50,  # number of itration to train on all data
        "log_interval": 10,  # interval between verbose on scree in iterations
        "validate": True,  # Whether to perform validation during training
        "loss": {
            "name": "WeightedMSELoss",
            "param": {"weight_fun": "linear", "C": 0.5, "y0": 0.1},
        },  # loss type for training
        "checkpoint_dir": "./DATA/checkpoints",  # Directory to save checkpoints
    },
}
