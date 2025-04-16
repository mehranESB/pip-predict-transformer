# Training Configuration

This document provides a detailed explanation of the training configuration options for the Trainer class. The configuration is defined as a Python dictionary, enabling users to customize every aspect of the training process, from dataset preparation to model optimization. Below is a breakdown of the configuration fields and their purposes.

## Overview

The training process is managed by a `Trainer` class, which handles the entire pipeline, including:

- Loading and augmenting datasets.
- Building the model.
- Configuring optimizers and schedulers.
- Managing checkpoints and reproducibility.
- Logging training progress.
- Users can modify the configuration dictionary to suit their specific requirements. The default configuration is as follows:
```python
config = {
    ...
}
```

## 1. Training Configuration

The `"train"` section defines parameters related to the training process itself. Example: 
```python
"train": {
    "device": "cuda",  
    "batch_size": 64,  
    "shuffle": True,  
    "scheduler": {
            ...
        }
    },
    "seed": 42,  
    "epochs": 50,  
    "log_interval": 10,  
    "loss_weight_fcn": "1.0",
    "validate": True,  
    "checkpoint_dir": "./DATA/checkpoints",  
}
```
| Field             | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `device`          | Specifies the computation device: `'cuda'` for GPU or `'cpu'` for CPU.     |
| `batch_size`      | Number of samples per batch for training.                                  |
| `shuffle`         | Whether to shuffle the training dataset during loading.                   |
| `scheduler`       | Learning rate scheduler configuration. See below for details.             |
| `seed`            | Random seed for ensuring reproducibility.                                 |
| `epochs`          | Total number of training epochs.                                          |
| `log_interval`    | Interval (in iterations) for logging metrics during training.             |
| `loss_weight_fcn` | the string of function to return weight for mse loss in specific score.             |
| `validate`        | Whether to validate the model after each epoch.                           |
| `checkpoint_dir`  | Directory to save training checkpoints.                                   |

### Scheduler Configuration

The `"scheduler"` field defines the learning rate adjustment strategy during training. Example:
```python
"scheduler": {
    "type": "StepLR",
    "parameters": {
        "step_size": 30,
        "gamma": 0.1
    }
}
```
| Field             | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `type`            | Scheduler type (e.g., StepLR, CosineAnnealingLR).                          |
| `parameters`      | Arguments specific to the chosen scheduler.                                |

## 2. Optimizer Configuration

The `"optimizer"` section configures the optimization algorithm. Example:
```python
"optimizer": {
    "type": "Adam",
    "parameters": {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0.0,
        "amsgrad": False
    },
    "load_from_checkpoint": False,
    "checkpoint_path": None
}
```
| Field             | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
|`type`	            |Optimizer type (e.g., Adam, SGD).                            |
|`parameters`	        |Configuration parameters for the optimizer.                  |
|`load_from_checkpoint` |	Whether to load an optimizer state from a checkpoint.     |
|`checkpoint_path`	|Path to the checkpoint file (if applicable).                 |

## 3. Model Configuration

The "model" section defines the architecture of the model. Example:
```python
"model": {
    "parameters": {
        "in_channels": 5,
        "d_model": 32,
        "num_blocks": 2,
        "num_layers": 2,
        "num_groups": 4,
        "embed_act_fun": "tanh",
        "act_fun": "relu",
        "nhead": 4,
        "num_encoder_layers": 4,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    "load_from_checkpoint": False,
    "checkpoint_path": None
}
```
| Field             | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
|`parameters`	    | Hyperparameters for building the model.                             |
|`load_from_checkpoint` |	Whether to initialize weights from a pretrained checkpoint.   |
|`checkpoint_path`  | Path to the checkpoint file (if applicable).                        |

## 4. Dataset Configuration

The `"data"` section specifies dataset paths, transformations, and splits. Example:
```python
"data": {
    "pkl_pathes": [
        "./DATA/pip/EURUSD-1h.pkl",
        "./DATA/pip/EURUSD-15m.pkl",
    ],
    "transformation": [
            ...
    ],
    "train_ratio": 0.8,   
    "valid_ratio": 0.15,  
}
```
| Field             | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
|`pkl_pathes`	    | List of paths to .pkl files containing the dataset.     |
|`transformation`	| List of transformations applied to the dataset.         |
|`train_ratio`	    | Proportion of the dataset used for training.            |
|`valid_ratio`	    | Proportion of the dataset used for validation.          |

### Transformation Configuration

Transformations augment the dataset to improve model generalization. Example:
```python
"transformation": [
    {
        "name": "mirror_reflect",
        "param": {"mode": "random"}
    },
    {
        "name": "add_uniform_score",
        "param": {
            "mu": 0.0235822,
            "sigma": 0.0227505,
            "xi": 0.652834
        }
    }
]
```
| Field             | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
|`mirror_reflect`	| Reflects the data horizontally, vertically, or both, based on the mode parameter. |
|`add_uniform_score` | Adds transformed scores to the dataset based on GEV distribution parameters (`mu`, `sigma`, `xi`). |

This configuration file is flexible and allows users to customize their training pipeline easily for different datasets and model architectures.