from pipdet.pipgen import generate_pip
from pathlib import Path

# Path to the CSV file containing the market OHLCV data
csv_path = Path("./DATA/csv/EURUSD-1h.csv")

# Path where the generated PIP data will be saved (in .pkl format)
save_path = Path("./DATA/pip/EURUSD-1h.pkl")
save_path.parent.mkdir(parents=True, exist_ok=True)

# Generate the PIP points and distances
generate_pip(
    csv_path,
    pip_info_to_save=[
        "dist",
        "hilo",
        "iter",
    ],  # these 3 column is necessary for training
    seq_len=128,
    inc_index=20,
    dist_method="perpendicular",
    save_path=save_path,
)

print(f"PIP data saved successfully to {save_path}")
