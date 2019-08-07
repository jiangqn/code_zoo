import torch
from torch import nn

class BinaryFocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=2.0, ignore_index=None, eps=1e-4):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, probs, labels):
        if self.ignore_index != None:
            mask = labels != self.ignore_index
        labels = labels.float()
        eps = torch.tensor(self.eps).to(labels.device)
        loss = - self.alpha * labels * torch.log(probs + eps) * (1 - probs).pow(self.gamma) - \
               (1 - self.alpha) * (1 - labels) * torch.log(1 - eps - probs) * probs.pow(self.gamma)
        if self.ignore_index != None:
            loss = loss.masked_select(mask)
        loss = loss.mean()
        return loss