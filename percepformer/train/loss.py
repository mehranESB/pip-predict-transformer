import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes Mean Squared Error loss between predict and target.

        Args:
            predict (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Scalar tensor representing MSE loss.
        """
        return torch.mean((predict - target) ** 2)


class WeightedMSELoss(nn.Module):
    def __init__(self, weight_fun: str = "linear", C: float = 1.0, y0: float = 1.0):
        """
        Initializes the weighted MSE loss.

        Args:
            weight_fun (str): Type of weighting function ("linear", "square", "exp").
            C (float): Scaling constant for weighting function.
            y0 (float): Offset for weighting function.
        """
        super(WeightedMSELoss, self).__init__()

        if weight_fun == "linear":
            self.weight_fun = lambda x: C * x + y0
        elif weight_fun == "square":
            self.weight_fun = lambda x: C * x**2 + y0
        elif weight_fun == "exp":
            self.weight_fun = lambda x: C * torch.exp(x) + y0
        else:
            raise ValueError(f"Unknown weight_fun: {weight_fun}")

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the weighted MSE loss.

        Args:
            predict (torch.Tensor): Predicted values.
            target (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        error = (predict - target) ** 2
        weights = self.weight_fun(target)
        weighted_loss = weights * error
        return torch.mean(weighted_loss)


class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        y_pred: Tensor of shape (batch_size, n_items)
        y_true: Tensor of same shape, real-valued scores

        Both should be raw scores (not ranks). The loss will convert them to
        probability distributions using softmax.
        """
        # Apply softmax to get probability distributions over permutations
        P_pred = F.softmax(y_pred, dim=1)
        P_true = F.softmax(y_true, dim=1)

        # Cross entropy loss between distributions
        loss = -torch.sum(P_true * torch.log(P_pred + 1e-10), dim=1)  # sum over items
        return loss.mean()  # mean over batch


def lossFcn(cfg):
    """
    Factory function to create and return a loss function object based on the provided configuration.
    Args:
        cfg (dict): A dictionary containing the configuration for the loss function.
    Returns:
        object: An instance of the specified loss function.
    Raises:
        ValueError: If the "name" key in the configuration does not match any known loss function.
    """

    if cfg["name"] == "MSELoss":
        return MSELoss()
    elif cfg["name"] == "WeightedMSELoss":
        return WeightedMSELoss(**cfg["param"])
    elif cfg["name"] == "ListNetLoss":
        return ListNetLoss()
    else:
        raise ValueError(f"Unknown loss function: {cfg['name']}")
