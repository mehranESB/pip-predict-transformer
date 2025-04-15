import logging

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
        # self.train_ds, self.valid_ds, self.test_ds = self.load_dataset(
        #     config)  # Load datasets
        # self.model = self.load_model(config)  # Load model
        # self.optimizer = self.load_optimizer(config)  # Load optimizer
        # self.prepare_to_train()  # prepare essentials to train
        # self.plotter = None  # plotter object to plot training process

        # # set random seed for reproducibility
        # self.set_seed()
