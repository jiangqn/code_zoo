import torch
from torch import nn

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        if self.ignore_index != None:
            mask = labels != self.ignore_index
            labels = torch.max(labels, torch.tensor(0).to(labels.device))
        one_hot = torch.zeros_like(logits).to(logits.device)
        one_hot = one_hot.scatter(1, labels.unsqueeze(-1), 1)
        prob = torch.softmax(logits, dim=-1)
        loss = - one_hot * torch.log(prob) * (1 - prob).pow(self.gamma)
        loss = loss.sum(dim=1, keepdim=False)
        if self.ignore_index != None:
            loss = loss.masked_select(mask)
        loss = loss.mean()
        return loss