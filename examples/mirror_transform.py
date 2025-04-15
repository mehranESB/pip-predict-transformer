from pipdet.dataset import PipDataset
from pipdet.utils import plot_pip_score
from percepformer.utils.transform import mirror_reflect
import matplotlib.pyplot as plt
from pathlib import Path


# Define the path to the dataset
pkl_path = Path("./DATA/pip/EURUSD-1h.pkl")

# create pip dataset and retrieve a sample
ds = PipDataset(pkl_path)
df_sample = ds[100]

# Plotting setup
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
(ax0, ax1, ax2, ax3) = axs.flatten()
sz = 50

# Original data
plot_pip_score(
    mirror_reflect(df_sample, mode="none"),
    title="Original Data",
    ax=ax0,
    max_markersize=sz,
)

# Vertical reflection
plot_pip_score(
    mirror_reflect(df_sample, mode="vertical"),
    title="Vertical Reflection",
    ax=ax1,
    max_markersize=sz,
)

# Horizontal reflection
plot_pip_score(
    mirror_reflect(df_sample, mode="horizontal"),
    title="Horizontal Reflection",
    ax=ax2,
    max_markersize=sz,
)

# Vertical-horizontal reflection
plot_pip_score(
    mirror_reflect(df_sample, mode="vertical-horizontal"),
    title="Vertical-Horizontal Reflection",
    ax=ax3,
    max_markersize=sz,
)


# Adjust layout and show the plot
plt.tight_layout()
plt.show()
