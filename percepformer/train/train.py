from .dataset import Dataset
from .loss import lossFcn
from ..model.model import create_model
from pipdet.dataset import PipDataset, CombinedDataset
import logging
from pathlib import Path
import torch
import random
import numpy as np
from .visual import TrainingPlotter

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],  # Output to console
)


class Trainer:

    def __init__(self, config: dict):
        """
        Initializes the Trainer class, loads datasets, the model, and optimizer.

        Args:
            config (dict): Configuration dictionary containing settings for the dataset, model, and optimizer.
        """
        self.logger = logging.getLogger(__name__)

        self.config = config  # Store the configuration

        # set random seed for reproducibility
        self.set_seed()

        # Load datasets
        self.train_ds, self.valid_ds, self.test_ds = self.load_dataset(config)

        # Load model
        self.model = self.load_model(config)

        # Load optimizer
        self.optimizer = self.load_optimizer(config)  # Load optimizer

        # prepare essentials to train
        self.prepare_to_train()

        # self.plotter = None  # plotter object to plot training process

    def set_seed(self):
        """
        Set the random seed for all relevant libraries to ensure reproducibility.
        This affects:
        - Python's random module
        - NumPy
        - PyTorch (CPU and CUDA)
        """

        # Retrieve the seed from the config, defaulting to 42 if not provided
        seed = self.config["train"].get("seed", 42)

        # Set the random seed for Python's random library
        random.seed(seed)

        # Set the random seed for NumPy
        np.random.seed(seed)

        # Set the random seed for PyTorch (CPU)
        torch.manual_seed(seed)

        # Set the random seed for CUDA (if using GPU)
        torch.cuda.manual_seed(seed)

        # Set the random seed for all GPUs (if using multiple GPUs)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in CUDA (to maintain reproducibility)
        torch.backends.cudnn.deterministic = True

        # Disable cuDNN auto-tuner to ensure deterministic results (can affect performance)
        torch.backends.cudnn.benchmark = False

    def load_dataset(self, config):
        """
        Load datasets based on the configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Tuple: train_ds, valid_ds, test_ds datasets.
        """
        # load datasets
        dataset_list = []
        for path in config["data"]["pkl_pathes"]:
            dataset_list.append(PipDataset(Path(path)))
            self.logger.info(f"Successfully imported Pip dataset from: {path}")

        # add dataset necessary informations to config
        ds = dataset_list[0]
        update_data = {
            "seq_len": getattr(ds, "seq_len"),
            "norm_width": getattr(ds, "norm_width"),
            "norm_height": getattr(ds, "seq_len"),
        }
        self.config["data"].update({"dataset": update_data})

        # Combine datasets
        combined_dataset = CombinedDataset(dataset_list)

        # Validate split ratios
        train_ratio = config["data"]["train_ratio"]
        valid_ratio = config["data"]["valid_ratio"]
        if train_ratio + valid_ratio > 1.0:
            raise ValueError("Sum of train_ratio and valid_ratio must not exceed 1.0.")

        # Split combined dataset
        train_ds, valid_ds, test_ds = combined_dataset.split(train_ratio, valid_ratio)
        self.logger.info("All Pip datasets have been successfully imported.")

        # Convert the split datasets into Dataset objects for proper data retrieval
        cfg = self.config["data"]["transformation"]
        train_ds = Dataset(train_ds, cfg)
        valid_ds = Dataset(valid_ds, cfg)
        test_ds = Dataset(test_ds, cfg)

        return train_ds, valid_ds, test_ds

    def load_model(self, config):
        """
        Load the model based on the provided configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters and checkpoint details.

        Returns:
            nn.Module: The instantiated and optionally pre-trained model.
        """
        # Create model from class
        model = create_model(**config["model"]["parameters"])

        # get number of model parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_params_in_million = num_params / 1e6  # Convert to millions
        self.logger.info(f"Model created with {num_params_in_million:.2f}M parameters.")

        # Load from a checkpoint if specified
        load_from_checkpoint = config["model"].get("load_from_checkpoint", False)
        checkpoint_path = config["model"].get("checkpoint_path", None)

        if load_from_checkpoint:
            if not checkpoint_path:
                raise ValueError(
                    "Checkpoint path must be provided when load_from_checkpoint is True."
                )

            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Weights path not found: {checkpoint_path}")

            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path)
            if "model_state_dict" not in checkpoint:
                raise KeyError("Checkpoint file does not contain 'model_state_dict'.")

            model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(
                f"Model weights loaded successfully from {checkpoint_path}"
            )

        return model

    def load_optimizer(self, config):
        """
        Initializes the optimizer for training based on the configuration.
        Optionally loads the optimizer state from a checkpoint.

        Args:
            config (dict): Configuration dictionary containing optimizer settings.

        Returns:
            torch.optim.Optimizer: Initialized optimizer instance.
        """
        # Extract optimizer type and parameters
        optimizer_type = config["optimizer"].get(
            "type", "Adam"
        )  # Default to Adam if not specified
        optimizer_params = config["optimizer"].get(
            "parameters", {"lr": 0.001}
        )  # Default parameters

        # Map optimizer type to the corresponding PyTorch optimizer
        optimizer_cls = getattr(torch.optim, optimizer_type, None)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Instantiate the optimizer
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        self.logger.info(
            f"Optimizer of type {type(optimizer).__name__} successfully initialized."
        )

        # Check if optimizer should be loaded from a checkpoint
        if config["optimizer"].get("load_from_checkpoint", False):
            checkpoint_path = Path(config["optimizer"].get("checkpoint_path", ""))
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Optimizer checkpoint not found: {checkpoint_path}"
                )

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            if "optimizer_state_dict" not in checkpoint:
                raise KeyError(
                    f"'optimizer_state_dict' not found in checkpoint at {checkpoint_path}"
                )

            # Load optimizer state
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.logger.info(
                f"Optimizer state loaded from checkpoint at: {checkpoint_path}"
            )

        return optimizer

    def prepare_to_train(self):
        """
        Prepare components required for the training loop.

        This includes setting up DataLoaders, moving the model to the specified device,
        and initializing the loss function.
        """
        # Training device (e.g., CPU or GPU)
        device_cfg = self.config["train"].get("device", "cpu")
        if "cuda" in device_cfg and torch.cuda.is_available():
            self.device = torch.device(device_cfg)
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.logger.info(
            f"Working device is {self.device} and model has been transferred to {self.device}."
        )

        # Create DataLoaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config["train"].get("batch_size", 32),
            shuffle=self.config["train"].get("shuffle", True),
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.config["train"].get("batch_size", 32),
            shuffle=False,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.config["train"].get("batch_size", 32),
            shuffle=False,
        )

        # create loss objective
        self.criterion = lossFcn(self.config["train"].get("loss", {"name": "MSELoss"}))

        # Scheduler (optional)
        if "scheduler" in self.config["train"]:
            scheduler_type = self.config["train"]["scheduler"].get("type", "StepLR")
            scheduler_params = self.config["train"]["scheduler"].get("parameters", {})
            scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_type, None)
            if scheduler_cls is None:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None

    def train(self):
        """
        Main training loop for the model.

        This includes iterating over epochs, training on batches, and optionally validating the model.
        """
        num_epochs = self.config["train"].get("epochs", 10)
        log_interval = self.config["train"].get("log_interval", 10)
        validate = self.config["train"].get("validate", True)

        # initialize iteration step
        iteration = 0
        self.plotter = TrainingPlotter()  # initialize plotter
        minimum_valid_loss = float("inf")  # store best validation loss ever

        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):

                inputs = batch["input"].to(self.device)
                targets = self.prepare_target(batch)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                iteration += 1

                if batch_idx % log_interval == 0:
                    self.plotter.update(iteration, train_loss=loss.item())
                    self.logger.info(
                        f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                    )

            # Average loss for the epoch
            epoch_loss = running_loss / len(self.train_loader)
            self.logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {epoch_loss:.4f}"
            )

            # Validation step
            if validate:
                valid_loss = self.validate()
                self.plotter.update(iteration, val_loss=valid_loss)
                minimum_valid_loss = (
                    valid_loss
                    if valid_loss < minimum_valid_loss
                    else minimum_valid_loss
                )

            # save all data and plot of losses
            self.save_checkpoint(epoch)
            self.save_plot(epoch)

            # Step scheduler if available
            if self.scheduler:
                self.scheduler.step()
                self.logger.info(
                    f"Learning rate adjusted to {self.optimizer.param_groups[0]['lr']} using the scheduler."
                )

        # close traiing process figure
        self.plotter.close_fig()

        # returning best validation loss
        return minimum_valid_loss

    def prepare_target(self, batch):
        """
        Prepare the target tensor for training, based on whether
        we're performing learning-to-rank (LTR) or standard regression.

        Args:
            batch (dict): Batch dictionary containing input tensors.

        Returns:
            target (Tensor): Prepared target tensor, on the correct device.
        """

        if self.config["model"].get("learn_to_sort", False):  # Learning-to-rank mode
            # Get indices that would sort each row in descending order
            iteration = batch["iter"]
            target = torch.argsort(iteration, dim=1, descending=True)

        else:  # Regression or metric-based mode

            # "udist" contains a uniforemed distance or score, and "hilo" indicates direction (0: low, 1: high)
            udist = batch["udist"]  # shape: [batch_size, n_items]
            hilo = batch["hilo"]  # binary tensor: 0 or 1

            # apply direction-aware transformation
            target = udist * (2.0 * hilo - 1.0)

        # Move target to correct device (e.g., GPU)
        return target.to(self.device)

    def validate(self):
        """
        Perform validation on the validation dataset.
        """

        self.logger.info("Starting validation loss computation.")
        self.model.eval()  # Set the model to evaluation mode
        validation_loss = 0.0

        with torch.no_grad():
            for batch in self.valid_loader:

                inputs = batch["input"].to(self.device)
                targets = self.prepare_target(batch)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                validation_loss += loss.item()

        # Average validation loss
        validation_loss /= len(self.valid_loader)
        self.logger.info(f"Validation Loss: {validation_loss:.4f}")

        return validation_loss

    def save_checkpoint(self, epoch):
        """
        Save the model, optimizer, and configuration to a checkpoint file.

        Args:
            epoch (int): The current epoch.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        checkpoint_dir = Path(
            self.config["train"].get("checkpoint_dir", "./checkpoints")
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")

    def save_plot(self, epoch):
        """
        save plot of training process.
        """

        # flag to save losses as json file at the end of training
        num_epochs = self.config["train"].get("epochs", 10)
        json_save = True if epoch >= (num_epochs - 1) else False

        # save and close
        checkpoint_dir = Path(
            self.config["train"].get("checkpoint_dir", "./checkpoints")
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = checkpoint_dir / f"loss_plot_epoch_{epoch}.png"
        self.plotter.save_plot(save_path, json_save=json_save)

        self.logger.info(
            f"Plot of training process for epoch {epoch} has been saved at {save_path}."
        )
