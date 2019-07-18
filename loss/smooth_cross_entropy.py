import torch
from torch import nn
import torch.nn.functional as F

class SmoothCrossEntropy(nn.Module):

    def __init__(self, smooth=0.1):
        super(SmoothCrossEntropy, self).__init__()
        # self.kldiv = nn.KLDivLoss()
        self.smooth = smooth

    def forward(self, input, target):
        one_hot = torch.zeros_like(input).to(input.device)
        one_hot = one_hot.scatter(1, target.unsqueeze(-1), 1)
        target = (1 - self.smooth) * one_hot + self.smooth / (input.size(1) - 1) * (1 - one_hot)
        input = input - input.max(dim=1, keepdim=True)[0]
        loss = -target * F.log_softmax(input, dim=-1)
        return loss.mean()