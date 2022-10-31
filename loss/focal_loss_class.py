
from sklearn.preprocessing import scale
import torch
from torch import instance_norm, nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean", *args, **kwargs):
        super().__init__()
        if weight:
            assert isinstance(weight,list)
            self.weight=torch.Tensor(weight).cuda()
        else:
            self.weight=None
        self.gamma = gamma
        assert reduction in ["sum", "mean", "none"]
        self.reduction = reduction

    def forward(self,input, target):
        loss=self._forward_one_mask(input[0],target)
        loss_aux=self._forward_one_mask(input[1],target)
        return loss+loss_aux

    def _forward_one_mask(self, input, target):
        target=target.squeeze(1)
        target=torch.cuda.LongTensor(target.cpu().numpy())
        loss = F.cross_entropy(input, target, reduction="none").view(-1)
        p = torch.exp(-loss)
        weights = self.weight[target.view(-1)] if self.weight is not None else torch.ones_like(target).view(-1)
        focal = weights * ((1 - p) ** self.gamma)
        if self.reduction == "mean":
            return (focal * loss).sum() / weights.sum()
        elif self.reduction == "sum":
            return (focal * loss).sum()
        else:
            return focal * loss

