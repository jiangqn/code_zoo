import torch
from torch import nn

class CapsuleLoss(nn.Module):

    def __init__(self, lambd=0.5, m1=0.9, m0=0.1):
        super(CapsuleLoss, self).__init__()
        self.lambd = lambd
        self.m1 = m1
        self.m0 = m0

    def forward(self, input, target):
        one_hot = torch.zeros_like(input).to(input.device)
        one_hot = one_hot.scatter(1, target.unsqueeze(-1), 1)
        zero = torch.zeros_like(input).to(input.device)
        a = torch.max(zero, self.m1 - input)
        b = torch.max(zero, input - self.m0)
        loss = one_hot * a * a + self.lambd * (1 - one_hot) * b * b
        loss = loss.sum(dim=1, keepdim=False)
        return loss.mean()