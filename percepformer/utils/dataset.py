from pipdet.dataset import PipDataset, CombinedDataset
from scipy.stats import genextreme
import numpy as np
from tqdm import tqdm


def dist_GEV_param(pkl_paths: list[str], sample_num: int = 10_000):
    """
    Fits a Generalized Extreme Value (GEV) distribution to the 'dist' scores from a list of PipDataset .pkl files.

    Args:
        pkl_paths (list[str]): List of paths to .pkl files containing the datasets.
        sample_num (int): Number of random samples to draw from the combined dataset for GEV fitting.

    Returns:
        tuple: Fitted GEV parameters (loc, scale, shape) in the order (mu, sigma, xi).
    """

    # Load PipDatasets
    pip_datasets = []
    for path in pkl_paths:
        ds = PipDataset(path)
        pip_datasets.append(ds)

    # Combine datasets into a single CombinedDataset
    combined_dataset = CombinedDataset(pip_datasets)

    # Randomly sample indices from the combined dataset
    sample_size = min(sample_num, len(combined_dataset))
    sample_indices = np.random.choice(
        range(len(combined_dataset)), size=sample_num, replace=False
    )

    # Collect 'dist' values from the sampled data
    dists_list = []
    for idx in tqdm(sample_indices, desc="Sampling 'dist' values"):
        df_sample = combined_dataset[idx]  # Unpack the data tuple
        dist = df_sample["dist"].to_numpy()
        dists_list.append(dist)

    # Concatenate all dist value into a single NumPy array
    dists = np.concatenate(dists_list)

    # Fit the GEV distribution to the data
    # Note: genextreme uses a slightly different parameterization.
    shape, loc, scale = genextreme.fit(dists)

    # Print the fitted parameters
    print(f"Shape parameter (xi): {-shape}")
    print(f"Location parameter (mu): {loc}")
    print(f"Scale parameter (sigma): {scale}")

    return (dists, loc, scale, -shape)  # Return (score, mu, sigma, xi)
