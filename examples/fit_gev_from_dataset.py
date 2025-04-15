from percepformer.utils.dataset import (
    dist_GEV_param,
)  # Adjust the import path to where your function is defined
from pathlib import Path

# List of paths to your .pkl files
pkl_files = [
    Path("./DATA/pip/EURUSD-15m.pkl"),
    Path("./DATA/pip/EURUSD-30m.pkl"),
    Path("./DATA/pip/EURUSD-1h.pkl"),
]

# Fit the GEV distribution using the provided function
dists, mu, sigma, xi = dist_GEV_param(pkl_files, sample_num=5000)
