from .train.train import Trainer
from .infer.inference import Detector
from .train.config import config


__all__ = ["config", "Trainer", "Detector"]
