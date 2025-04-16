# Data Preprocessing Guide

This guide explains how to preprocess raw financial data for training the model. The preprocessing involves normalizing data to match a trader's perspective, ensuring compatibility with the PIP detection process. Below are the detailed steps.

## Normalizing Data

To make the data visually intuitive and consistent, it is normalized in both the x (time) and y (price) axes. This normalization ensures the data aligns with a trader's eye view, with a fixed width and height. The normalization process is illustrated with an example.

### Example Code for Normalization

Here’s a step-by-step example of how to normalize data:
```python
from multitimeprep.utils import csv as csv_utils
from multitimeprep.dataset import SingleDataset
from percepformer.utils.transform import tight_box_normalize
from pathlib import Path

# Import source DataFrame
csv_source_path = Path("./DATA/csv/EURUSD-15m.csv")
df_source = csv_utils.import_ohlcv_from_csv(
    csv_source_path, header=True, datetime_format="%Y-%m-%d %H:%M:%S"
)

# Create dataset and fetch a sample
dataset = SingleDataset(df_source, 128)  # 128 defines the number of data points per sample
sample_data = dataset[100]  # Fetch the 100th sample

# Normalize the sample data with 
normalized_data = tight_box_normalize(sample_data, width=1.6, height=1.0)
```

`sample_data` is the original OHLCV (Open, High, Low, Close, Volume) data extracted from CSV files, and `normalized_data` is the scaled version of `sample_data` after applying `tight_box_normalize`, fitting it into a consistent visual format. 

![normalized_data](../images/normalized.png)

Above is a visual comparison of raw data and normalized data for better understanding.

## PIP Point Information Generation

To identify and extract perceptually important points (PIP) from the normalized data, we use the `FastPip` class from the `percepformer.utils.pip` module. The `FastPip` class computes various PIP feature informations of points iteratively, continuing this process until all points are evaluated (Normalization is applied automatically to the data).

The following features are generated for each point:
- `iter`: Indicating how many times `FastPip` iterates till consider the point as PIP point.
- `hilo`: This feature indicates whether the point represents a High (1.0) or Low (0.0) price of the bar, reflecting its perceptual importance from a trader's perspective.
- `dist`: The distance of the PIP point from its segment, calculated using the perpendicular method. This helps quantify how far the point is from the line connecting adjacent data points.
- **Other Features:** The FastPip class also computes additional features and other geometrical properties to provide a comprehensive understanding of each point's importance.

These features, along with the market data, are saved into `.pkl` files, allowing for easy access when creating the dataset.
### Example Code for Generating PIP Points
```python 
from percepformer.dataset.pipgen import generate_pip
from pathlib import Path

# Path to the CSV file containing the market OHLCV data
csv_path = Path("./DATA/csv/EURUSD-1h.csv")

# Path where the generated PIP data will be saved (in .pkl format)
save_path = Path("./DATA/pip/EURUSD-1h.pkl")

# Generate the PIP points and distances
generate_pip(csv_path, seq_len=128, inc_index=20,
             dist_method="perpendicular", save_path=save_path)
```

- `seq_len=128`: Specifies the sequence length for the ohlc data in each sample.
- `inc_index=20`: Defines the index increment to get next sample in dataset for generating PIP points.
- `dist_method="perpendicular"`: Chooses the distance calculation method for determining how far each point is from its segment.
- `save_path`: The location where the generated `.pkl` file will be saved.

The generated `.pkl` file will contain the PIP point features and the corresponding market data for further training steps.

![visualize_pip_points](../images/pip_visualize.png)

The figure above shows the result of identifying PIP points, where points are filtered out if their `dist` feature is lower than 0.1.

## Creating and Combining Datasets

Once the PIP point information is generated and saved into `.pkl` files, the next step is to create dataset objects that can be used for training, validation, and testing. This can be done using the `PipDataset` class, which automatically loads the dataset from the provided `.pkl` files. Multiple datasets can also be combined into a single dataset for broader data access during training.

### `PipDataset`

The `PipDataset` class loads a `.pkl` file containing PIP point data and prepares it for training. This object allows for easy access to PIP points and associated features.

### `CombinedDataset`

If you want to merge multiple `PipDataset` objects, you can use the `CombinedDataset` class. This class combines datasets into a single unified dataset, which is especially useful when you want to use data from different time frames or different market data sources.

### Example Code for Creating and Combining Datasets
```python
from percepformer.dataset.dataset import PipDataset, CombinedDataset
from pathlib import Path

# Define the paths to the .pkl files containing PipDatasets
pkl_path1 = Path("./DATA/pip/EURUSD-1h.pkl")
pkl_path2 = Path("./DATA/pip/EURUSD-15m.pkl")

# Load the PipDatasets
dataset1 = PipDataset(pkl_path1)  # Load the first PipDataset
dataset2 = PipDataset(pkl_path2)  # Load the second PipDataset

# Combine the PipDatasets into a single CombinedDataset
# Merge datasets for unified management
combined_dataset = CombinedDataset([dataset1, dataset2])

# Split the combined dataset into training (80%), validation (15%), and testing (5%)
# Assumes the `split` method of CombinedDataset is implemented
train_ds, valid_ds, test_ds = combined_dataset.split(0.8, 0.15)
```

### Dataset Splitting

Both `CombinedDataset` and `PipDataset` class also support dataset splitting. You can easily split the combined dataset into training, validation, and testing sets, as shown in the example. The split method divides the data based on the provided percentages (in our example: 80% for training, 15% for validation, and 5% for testing).

## Data Augmentation: Mirror Reflection

To enhance the dataset and increase its size, we apply data augmentation during the training process. One of the augmentation techniques used is mirror reflection, which reflects the market data along one or both axes (horizontal, vertical, or both). This technique allows the model to learn from more varied patterns by effectively generating new market data.

When using the mirror_reflect transformation, we can choose between different modes:
- **Horizontal Reflection:** Reflects the data along the vertical axis.
- **Vertical Reflection:** Reflects the data along the horizontal axis.
- **Vertical-Horizontal Reflection:** Reflects the data along both vertical and horizontal axes.
- **Random Mode:** Randomly selects between horizontal, vertical, both and no-change reflections for each data sample.

By applying the random mode, the dataset becomes four times larger, as each data point can be transformed in any of the four configurations (original, horizontally reflected, vertically reflected, and both). This random augmentation helps improve the model's robustness by providing a wider variety of market data during training.

### Example Code for Mirror Reflection Augmentation
```python
from percepformer.dataset.dataset import PipDataset
from percepformer.utils.transform import mirror_reflect
from pathlib import Path

# Define the path to the dataset
pkl_path = Path("./DATA/pip/EURUSD-1h.pkl")

# Define transformations
T = [lambda x: mirror_reflect(x, mode="random")]  # Random mirror reflection

# Create PipDataset instance with transformations
dataset = PipDataset(pkl_path, transforms=T)
```

In this example:
-   The `mirror_reflect` transformation is applied with the `"random"` mode, so the dataset will randomly reflect data along either axis (horizontal, vertical, both or none).
-   The dataset is then loaded using the `PipDataset` class, with the transformations included.

This augmentation ensures that the dataset is larger and more diverse, leading to better generalization during training. 

![mirror_reflection](../images/mirror_reflect.png)

The figure above illustrates the market data after all transformations (including mirror reflections) have been applied. PIP points are overlaid on the data, with the size of each scatter point varying based on its importance. 

## Transforming Target Feature for Learning

In this project, we focus on learning the perceptual importance of PIP points. Among the features generated for PIP points, the `dist` feature plays a crucial role as it represents the importance of each point. This feature is treated as the score that the model aims to predict.

### Addressing Imbalanced Distribution

The raw `dist` values are often heavily imbalanced, with the majority of points having values close to zero. This imbalance poses a challenge for regression tasks, as using losses like Mean Squared Error (MSELoss) may lead to poor learning performance. To address this, we transform the `dist` values into a more balanced distribution.

### Transformation Process

1. **Fitting a Generalized Extreme Value (GEV) Distribution:**
    The `dist` values are modeled using a GEV distribution, which effectively captures the tail-heavy nature of the data.
2. **Applying the Cumulative Distribution Function (CDF):**
    The fitted GEV distribution's CDF is applied to the `dist` values, transforming them into a uniform distribution ranging from 0 to 1. This transformation balances the target scores, making them more suitable for training.
3. **Inverse Transformation After Training:**
    Once the model is trained, the predicted scores (transformed by the CDF) are mapped back to the original scale of `dist` by applying the inverse CDF function of the GEV distribution. This step ensures that the results are interpretable in terms of the original feature space.
    
### Setting Up GEV Parameters for Training

Before training the model, it is essential to compute the parameters of the Generalized Extreme Value (GEV) distribution for the `dist` feature. These parameters will be used to configure the target transformation for better learning performance. Use the `dist_GEV_param` function to calculate the parameters from your prepared datasets. Below is an example script:
```python 
from percepformer.dataset.pipgen import dist_GEV_param

# Define the paths to the .pkl files containing PipDatasets
pkl_path1 = "./DATA/pip/EURUSD-15m.pkl"
pkl_path2 = "./DATA/pip/EURUSD-30m.pkl"
pkl_path3 = "./DATA/pip/EURUSD-1h.pkl"

path_list = [pkl_path1, pkl_path2, pkl_path3]

# Calculate GEV distribution parameters
_, mu, sigma, xi = dist_GEV_param(path_list, sample_num=10_000, apply_transform=True)
```

The calculated values of `mu`, `sigma`, and `xi` (location, scale, and shape parameters) should be added to the training configuration. This ensures that the transformed target scores are properly scaled and uniform, facilitating efficient training.

![cdf_transform](../images/score_distribution.png)

The figure above illustrates the distribution of the original scores (`dist`) and their normalized counterparts (CDF(`dist`)) after applying the Generalized Extreme Value (GEV) cumulative distribution function.