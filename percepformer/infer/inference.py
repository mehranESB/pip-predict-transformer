from .load import load_checkpoint, load_model_from_checkpoint
import torch
from pipdet.utils import tight_box_normalize_df
import pandas as pd
import numpy as np
from ..utils.transform import gev_inverse_cdf


class Detector:

    def __init__(self, checkpoint_path):
        # Load the checkpoint from the given path
        chkpoint = load_checkpoint(checkpoint_path, "cpu")
        self.config = chkpoint["config"]

        # Load the model using the loaded checkpoint
        self.model = load_model_from_checkpoint(chkpoint)

        # Set device (CPU or CUDA) for inference
        self.set_device()

        # Set up the normalization function
        self.set_normalizer()

        # Set up the inverter from uniform distribution of dist to dist
        self.set_inverter()

    def __call__(
        self, data: pd.DataFrame, threshold: float = 0.0, points_num: int = None
    ):
        """
        Run inference on the input DataFrame.

        Args:
            data (pd.DataFrame): Input OHLC data with 'X' coordinate.
            threshold (float): Minimum value for filtering based on distance from seqment.
            points_num (int, optional): If set, selects only top-N predictions.

        Returns:
            pd.DataFrame: DataFrame with added prediction results.
        """
        # Normalize the input data
        data = self.normalizer(data)

        # Extract input features and prepare tensor
        input_data = (
            data[["Open", "High", "Low", "Close", "X"]].to_numpy().astype(np.float32)
        )
        input_tensor = (
            torch.tensor(input_data).to(self.device).unsqueeze(0)
        )  # Add batch dim

        # Run inference
        with torch.no_grad():
            result = self.model(input_tensor)[0]  # Assuming output shape is [1, N]

        # Move result to CPU and convert to NumPy
        result = result.cpu().numpy()

        # Determine direction
        hilo = (result > 0) * 1.0 + (result < 0) * 0.0 + (result == 0) * 0.5

        # Absolute distance and inversion
        udist = np.abs(result)
        dist = np.abs(self.inverter(udist))

        # Prepare result dictionary
        pip_results = {
            "hilo": hilo,
            "udist": udist,
            "dist": dist,
            "is": dist >= threshold,
            "coordX": data["X"].to_numpy(),
            "coordY": hilo * data["High"].to_numpy()
            + (1 - hilo) * data["Low"].to_numpy(),
        }

        # Sort indices by descending distance
        sort_idx = np.argsort(-dist)
        iter_idx = np.zeros_like(sort_idx)
        iter_idx[sort_idx] = np.arange(len(sort_idx))
        pip_results["iter"] = iter_idx

        # Apply top-N filtering if points_num is set
        if points_num is not None:
            selected = sort_idx[:points_num]
            is_selected = np.zeros_like(result, dtype=bool)
            is_selected[selected] = True
            is_selected[[0, -1]] = True
            pip_results["is"] = is_selected

        # Add results back to the DataFrame
        for key, val in pip_results.items():
            data[key] = val

        return data

    def set_device(self):
        """
        Configure the device (CPU or GPU) for the model based on the config.
        """
        device_cfg = self.config["train"].get("device", "cpu")

        # Use CUDA if available and specified
        if "cuda" in device_cfg and torch.cuda.is_available():
            self.device = torch.device(device_cfg)
        else:
            self.device = torch.device("cpu")

        # Move model to the selected device
        self.model = self.model.to(self.device)

    def set_normalizer(self):
        """
        Prepare a normalization function using the dataset config.
        """
        dataset_cfg = self.config["data"]["dataset"]

        norm_width = dataset_cfg["norm_width"]
        norm_height = dataset_cfg["norm_height"]

        # Create a normalization lambda using given dimensions
        self.normalizer = lambda x: tight_box_normalize_df(
            x, width=norm_width, height=norm_height
        )

    def set_inverter(self):
        """
        Create an inverter function using the GEV inverse CDF
        with parameters specified in the transformation config.

        Supports both old (list) and new (dict) config formats.
        """
        trans_cfg = self.config["data"]["transformation"]
        uniform_param = None

        # Old format: list of transformation steps
        if isinstance(trans_cfg, list):
            for trns in trans_cfg:
                if trns.get("name") == "add_uniform_score":
                    uniform_param = trns.get("param")
                    break

        # New format: transformation dictionary
        elif isinstance(trans_cfg, dict):
            uniform_param = trans_cfg.get("add_uniform_score")

        # Raise error if config is malformed or missing required parameters
        if uniform_param is None:
            raise ValueError(
                "Missing 'add_uniform_score' parameters in transformation config."
            )

        # Create inverter lambda function
        self.inverter = lambda x: gev_inverse_cdf(x, **uniform_param)
