from percepformer.train.config import config
from percepformer.train.train import Trainer

# --------------------------
# Training Configuration
# --------------------------
# Basic training configuration (batch size, number of epochs, checkpoint directory)
config["train"]["batch_size"] = 64  # Number of samples per batch
config["train"]["epochs"] = 50  # Total number of epochs for training
# Step size for scheduler (learning rate decay)
config["train"]["scheduler"]["parameters"]["step_size"] = 30
# Factor by which the learning rate will be reduced
config["train"]["scheduler"]["parameters"]["gamma"] = 0.1
# Directory to save model checkpoints
config["train"]["checkpoint_dir"] = "./DATA/checkpoints/train_0"

# ---------------------------
# Optimizer Configuration
# ---------------------------
config["optimizer"]["parameters"]["lr"] = 0.001  # Learning rate
# Weight decay (L2 regularization), set to 0 for no regularization
config["optimizer"]["parameters"]["weight_decay"] = 0.0

# ---------------------------
# Dataset Configuration
# ---------------------------
# Ratio of the data used for training (80% of the dataset)
config["data"]["train_ratio"] = 0.8
# Ratio of data used for validation (15% of the dataset)
config["data"]["valid_ratio"] = 0.15
config["data"]["pkl_paths"] = [  # Paths to your dataset pickle files (ensure they exist)
    "./DATA/pip/EURUSD-1h.pkl",  # 1-hour data
    "./DATA/pip/EURUSD-30m.pkl",  # 30-minute data
    "./DATA/pip/EURUSD-15m.pkl",  # 15-minute data
]

# ---------------------------
# Model Configuration
# ---------------------------
# Dimension of the model's output space (embedding size)
config["model"]["parameters"]["d_model"] = 32
config["model"]["parameters"]["num_layers"] = 2  # Number of layers in the model
# Number of groups for any group-based attention mechanism
config["model"]["parameters"]["num_groups"] = 4
# Activation function (ReLU here, but others can be used, e.g., 'gelu')
config["model"]["parameters"]["act_fun"] = "relu"
# Number of attention heads for multi-head attention
config["model"]["parameters"]["nhead"] = 4
config["model"]["parameters"]["num_encoder_layers"] = 4  # Number of encoder layers
# Dimension of the feedforward layer
config["model"]["parameters"]["dim_feedforward"] = 128
config["model"]["parameters"]["dropout"] = 0.1  # Dropout rate (helps prevent overfitting)

# ---------------------------
# Initialize the Trainer
# ---------------------------
trainer = Trainer(config)  # Initialize the Trainer class with the config

# ---------------------------
# Start the Training
# ---------------------------
trainer.train()  # Begin the training process
