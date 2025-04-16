from percepformer.train.config import config
from percepformer.train.train import Trainer
from percepformer.utils.load import load_checkpoint_model_config

# --------------------------
# Training Configuration
# --------------------------
# Basic training configuration (batch size, number of epochs, checkpoint directory)
config["train"]["batch_size"] = 64  # Number of samples per batch
config["train"]["epochs"] = 60  # Total number of epochs for training
# Step size for scheduler (learning rate decay)
config["train"]["scheduler"]["parameters"]["step_size"] = 60
# Factor by which the learning rate will be reduced
config["train"]["scheduler"]["parameters"]["gamma"] = 0.1
# Directory to save model checkpoints
config["train"]["checkpoint_dir"] = "./DATA/checkpoints/train_7_dropout"
config["train"]["score_threshold"] = None

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
config["data"]["train_ratio"] = 0.9
# Ratio of data used for validation (15% of the dataset)
config["data"]["valid_ratio"] = 0.1
config["data"]["pkl_pathes"] = [  # Paths to your dataset pickle files (ensure they exist)
    "./DATA/pip/EURUSD-1h.pkl",  # 1-hour data
    "./DATA/pip/EURUSD-30m.pkl",  # 30-minute data
    "./DATA/pip/EURUSD-15m.pkl",  # 15-minute data
]

# ---------------------------
# Load Pretrained
# ---------------------------
# Load model from checkpoint and update training configuration
try:
    checkpoint_path = "./DATA/checkpoints/train_7/checkpoint_epoch_59.pth"

    # Load the model configuration from the checkpoint
    model_config = load_checkpoint_model_config(checkpoint_path)

    # Update the overall training configuration with the model configuration
    config["model"] = model_config

    # Log success (if a logger is available)
    print("Model configuration successfully updated to load from checkpoint.")
except (FileNotFoundError, KeyError) as e:
    print(f"Error loading model configuration from checkpoint: {e}")

# ---------------------------
# Initialize the Trainer
# ---------------------------
trainer = Trainer(config)  # Initialize the Trainer class with the config

# ---------------------------
# Start the Training
# ---------------------------
trainer.train()  # Begin the training process
