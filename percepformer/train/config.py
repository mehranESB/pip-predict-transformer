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
}
