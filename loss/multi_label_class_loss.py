import numpy as np
import torch
import torch.nn as nn

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weights=None, pos_weight=None, **kwargs):
        """
        Args:
            pos_weight： 各個類別正樣本權重
            weight:  樣本權重
            **kwargs:
        """
        super(BCEWithLogitsLoss, self).__init__()
        self.criteria = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels)
        return loss


class SoftmaxCrossentropyLoss(nn.Module):
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs:
        """
        super(SoftmaxCrossentropyLoss, self).__init__()

    def forward(self, logits, labels):
        """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
             1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
             不用加激活函数，尤其是不能加sigmoid或者softmax！预测
             阶段则输出y_pred大于0的类
        """
        y_pred = (1 - 2 * labels) * logits
        y_pred_neg = y_pred - labels * 1e12
        y_pred_pos = y_pred - (1 - labels) * 1e12
        zeros_neg = torch.zeros_like(y_pred[..., :1]) + 0.5
        zeros_pos = torch.zeros_like(y_pred[..., :1]) - 0.5
        y_pred_neg = torch.cat([y_pred_neg, zeros_neg], axis=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros_pos], axis=-1)
        neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
        pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
        return (neg_loss + pos_loss).mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=2, gamma_pos=1, clip=0.01, eps=1e-8, disable_torch_grad_focal_loss=True,**kwargs):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
