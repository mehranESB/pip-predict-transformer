from .dataset import Dataset
from pipdet.dataset import PipDataset, CombinedDataset
import logging
from pathlib import Path

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

        # Load datasets
        self.train_ds, self.valid_ds, self.test_ds = self.load_dataset(config)

        # self.model = self.load_model(config)  # Load model
        # self.optimizer = self.load_optimizer(config)  # Load optimizer
        # self.prepare_to_train()  # prepare essentials to train
        # self.plotter = None  # plotter object to plot training process

        # # set random seed for reproducibility
        # self.set_seed()

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
