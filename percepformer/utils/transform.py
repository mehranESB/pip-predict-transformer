from scipy.stats import genextreme
import numpy as np
import pandas as pd
import random


def mirror_reflect(data: pd.DataFrame, mode: str = "random"):
    """
    Apply a specified or random mirroring transformation to market data.

    Args:
        data (pd.DataFrame): Input market data with columns
            ["Open", "High", "Low", "Close", "X", "dist", "hilo", "iter"].
        mode (str): The mode of reflection. Options are 'none', 'random',
            'vertical', 'horizontal', or 'vertical-horizontal'.

    Returns:
        pd.DataFrame: The transformed market data as a new DataFrame.
    """

    # Select OHLC and other columns properly (use df[[]] instead of df())
    ohlc_data = data[["Open", "High", "Low", "Close"]].to_numpy()
    other_data = data[["X", "dist", "hilo", "iter"]].to_numpy()

    # Unpack other data
    x, dist, hilo, iteration = (
        other_data[:, 0],
        other_data[:, 1],
        other_data[:, 2],
        other_data[:, 3],
    )

    if mode == "none":
        return data.copy()  # Just return a copy if no transformation is requested

    elif mode == "random":
        mode = random.choice(["none", "vertical", "horizontal", "vertical-horizontal"])
        return mirror_reflect(data, mode=mode)

    elif mode == "vertical":
        ohlc_data = np.flip(ohlc_data, axis=0)
        ohlc_data[:, [0, 3]] = ohlc_data[:, [3, 0]]  # Swap Open and Close

        dist = np.flip(dist)
        hilo = np.flip(hilo)
        iteration = np.flip(iteration)

    elif mode == "horizontal":
        ohlc_data = -1 * ohlc_data
        ohlc_data = ohlc_data - ohlc_data.min()  # Shift to make values non-negative
        ohlc_data[:, [1, 2]] = ohlc_data[:, [2, 1]]  # Swap High and Low
        hilo = 1.0 - hilo  # Flip HiLo flag

    elif mode == "vertical-horizontal":
        return mirror_reflect(mirror_reflect(data, mode="horizontal"), mode="vertical")

    else:
        raise ValueError(
            f"Invalid mode '{mode}' specified. Valid modes are 'none', 'random', 'vertical', 'horizontal', 'vertical-horizontal'."
        )

    # Reconstruct the DataFrame
    transformed_data = pd.DataFrame(
        np.column_stack([ohlc_data, x, dist, hilo, iteration]),
        columns=["Open", "High", "Low", "Close", "X", "dist", "hilo", "iter"],
    )

    return transformed_data


def add_uniform_score(data: pd.DataFrame, mu: float, sigma: float, xi: float):
    """
    Converts the "dist" column of the given DataFrame into uniformly distributed values
    using the GEV CDF and adds the result as a new column, "udist".

    Parameters:
        data (pd.DataFrame): A DataFrame with a "dist" column.
        mu (float): The location parameter of the GEV distribution.
        sigma (float): The scale parameter of the GEV distribution.
        xi (float): The shape parameter of the GEV distribution.

    Returns:
        pd.DataFrame: The input DataFrame with an added "udist" column.
    """
    if "dist" not in data.columns:
        raise ValueError("Input data must contain a 'dist' column.")

    dist = data["dist"].to_numpy()

    # Compute GEV CDF (note: SciPy uses -xi for shape)
    uniform_dist = genextreme.cdf(dist, -xi, loc=mu, scale=sigma)

    # Sanity check
    if not np.all((uniform_dist >= 0) & (uniform_dist <= 1)):
        raise ValueError(
            "Generated uniform scores are not in [0, 1]. Check GEV parameters."
        )

    # Add as new column
    data = data.copy()  # avoid modifying original DataFrame
    data["udist"] = uniform_dist

    return data


def gev_inverse_cdf(y, mu, sigma, xi):
    if np.any((y <= 0) | (y >= 1)):
        raise ValueError("Uniform values must be in (0, 1).")
    if sigma <= 0:
        raise ValueError("Scale parameter sigma must be positive.")

    return genextreme.ppf(y, -xi, loc=mu, scale=sigma)
