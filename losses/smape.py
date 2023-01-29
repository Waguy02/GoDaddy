import torch
from torch import nn


class SmapeCriterion(nn.Module):
    """
    Class to compute the SMAPE loss.
    """
    def __init__(self):
        super(SmapeCriterion, self).__init__()

    def forward(self, y_pred, y_true):
        """
        @param y_pred: Predicted values
        @param y_true: True values
        @return: SMAPE loss
        """
        eps = 1e-8
        return 100*torch.mean(2 * torch.abs(y_pred - y_true) / (torch.abs(y_pred) + torch.abs(y_true) + eps))

    def __str__(self):
        return "SMAPE"

    def __repr__(self):
        return str(self)

