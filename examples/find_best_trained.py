import os
import json
import torch
import matplotlib.pyplot as plt

# Path to the main "checkpoints" directory
CHECKPOINTS_DIR = "./DATA/checkpoints"

# Initialize lists for storing data
model_parameters_list = []
validation_losses_list = []
folder_names = []

# Iterate over all subdirectories in the checkpoints folder
for folder in os.listdir(CHECKPOINTS_DIR):
    folder_path = os.path.join(CHECKPOINTS_DIR, folder)

    if not os.path.isdir(folder_path):
        continue  # Skip if it's not a directory

    # Find the last checkpoint file
    checkpoint_files = [
        f for f in os.listdir(folder_path) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
    ]

    if not checkpoint_files:
        continue  # Skip if no checkpoint files found

    # Extract epoch numbers and find the latest
    epoch_numbers = [int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files]
    latest_epoch = max(epoch_numbers)
    latest_checkpoint_file = f"checkpoint_epoch_{latest_epoch}.pth"
    latest_checkpoint_path = os.path.join(folder_path, latest_checkpoint_file)

    # Load the checkpoint
    checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")

    # Extract model parameters count
    model_params = sum(p.numel() for p in checkpoint["model_state_dict"].values())
    model_parameters_list.append(model_params)

    # Load validation loss from JSON
    json_file = f"loss_plot_epoch_{latest_epoch}_losses.json"
    json_path = os.path.join(folder_path, json_file)

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            loss_data = json.load(f)
            val_loss = loss_data.get("val_losses", [])

            if val_loss:  # If val_loss is not empty, take the last recorded value
                validation_losses_list.append(val_loss[-1])
                folder_names.append(folder)  # Store folder name for labeling
            else:
                validation_losses_list.append(None)
                folder_names.append(None)
    else:
        validation_losses_list.append(None)
        folder_names.append(None)

# Filter out any None values (if some models lack validation losses)
filtered_data = [
    (params, loss, name) for params, loss, name in zip(model_parameters_list, validation_losses_list, folder_names) if loss is not None
]

# Unpack the filtered data
if filtered_data:
    model_parameters_list, validation_losses_list, folder_names = zip(*filtered_data)

    # Plot validation loss vs. model parameters
    plt.figure(figsize=(8, 6))
    plt.scatter(model_parameters_list, validation_losses_list,
                color="blue", label="Validation Loss")

    # Add folder names as labels above each point
    for param, loss, name in zip(model_parameters_list, validation_losses_list, folder_names):
        plt.text(param, loss, name, fontsize=9, ha='right', va='bottom', color='red')

    plt.xlabel("Number of Model Parameters")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs. Model Parameters")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No valid data found for plotting.")
