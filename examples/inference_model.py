from percepformer.infer.inference import Detector
import chartDL.utils.csv as csv_utils
from pathlib import Path
from pipdet.utils import plot_pip_on_chart, plot_pip_score
import matplotlib.pyplot as plt

# Load the model inference object with the checkpoint
checkpoint_path = Path("./DATA/best_checkpoints/medium/medium_1M.pth")
detector = Detector(checkpoint_path)

# Import OHLCV data from the CSV file as test input
csv_source_path = Path("./DATA/csv/EURUSD-1h.csv")
df_source = csv_utils.import_ohlcv_from_csv(
    csv_source_path, header=True, datetime_format="%Y-%m-%d %H:%M:%S"
)

# Select a specific sample for inference
start_idx = 0
seq_len = 128
df_sample = df_source.iloc[start_idx : start_idx + seq_len]

# inference
df_results = detector(df_sample, points_num=10)

# visualize
fig, ax = plt.subplots()
plot_pip_on_chart(df_results, title="predicted pip distance as score", ax=ax)
plot_pip_score(
    df_results, title="predicted pip distance as score", ax=ax, max_markersize=50
)
plt.show()
