import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1)
        targets_onehot = F.one_hot(targets, preds.shape[1]).permute(0,3,1,2).float()
        intersection = (preds * targets_onehot).sum(dim=(2,3))
        union = preds.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()