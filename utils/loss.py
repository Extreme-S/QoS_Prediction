import torch
import torch.nn as nn
import math

class CauchyLoss(nn.Module):
    def __init__(self, gamma=1):
        super(CauchyLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        diff = inputs - targets
        loss = torch.mean(0.5 * torch.log((self.gamma + diff ** 2) / self.gamma))
        return loss
