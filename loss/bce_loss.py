import torch
from torch import nn

class BCELoss(nn.Module):

    def __init__(self, ignore_index=None, eps=1e-4):
        super(BCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, probs, labels):
        if self.ignore_index != None:
            mask = labels != self.ignore_index
        labels = labels.float()
        eps = torch.tensor(self.eps).to(labels.device)
        loss = - labels * torch.log(probs + eps) - (1 - labels) * torch.log(1 - eps - probs)
        if self.ignore_index != None:
            loss = loss.masked_select(mask)
        loss = loss.mean()
        return loss