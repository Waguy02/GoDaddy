import torch
from torch import nn
from torchmetrics import SymmetricMeanAbsolutePercentageError


class SmapeCriterion(nn.Module):
    """
    Class to compute the SMAPE loss.
    """
    def __init__(self):
        super(SmapeCriterion, self).__init__()
        self.smape = SymmetricMeanAbsolutePercentageError()

    def forward(self, y_pred, y_true):
        """
        @param y_pred: Predicted values
        @param y_true: True values
        @return: SMAPE loss
        """
        return self.smape(y_pred, y_true)*100

    def __str__(self):
        return "SMAPE"

    def __repr__(self):
        return str(self)

