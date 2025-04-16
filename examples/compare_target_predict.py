from percepformer.infer.inference import Detector
from pipdet.dataset import PipDataset
from pathlib import Path
from pipdet.utils import plot_pip_score
import matplotlib.pyplot as plt

# Load the model inference object with the checkpoint
checkpoint_path = Path("./DATA/best_checkpoints/medium/medium_1M.pth")
detector = Detector(checkpoint_path)

# import a pip dataset
pip_ds_path = Path("./DATA/pip/EURUSD-1h.pkl")
dataset = PipDataset(pip_ds_path)

# sample data
data_sample = dataset[100]

# get the input part of data
data_input = data_sample[["TimeStamp", "Open", "High", "Low", "Close"]].copy()
data_out = detector(data_input)

# visualize
fig, (ax0, ax1) = plt.subplots(2, 1)
plot_pip_score(data_sample, title="Target", ax=ax0, max_markersize=50)
plot_pip_score(data_out, title="Predicted", ax=ax1, max_markersize=50)
plt.show()
