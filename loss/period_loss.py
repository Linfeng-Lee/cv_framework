import torch
import torch.nn as nn


class PeriodLoss(nn.Module):
    def __init__(self, period_thresh=0.7, period_n_min=1000, period_ignore_lb=-255,period_weights=None, *args, **kwargs):
        """
        Args:
            thresh: 判断困难像素的样本阈值
            n_min:  用于统计loss的像素数量
            ignore_lb:  忽略不统计loss的label
            weights: 。
            *args:
            **kwargs:
        """
        super(PeriodLoss, self).__init__()

        self.thresh = -torch.log(torch.tensor(period_thresh, dtype=torch.float)).cuda()
        self.n_min = period_n_min
        self.ignore_lb = period_ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=period_ignore_lb, reduction='none',weight=period_weights)

    def forward(self, logits, labels):
        aux_logits=logits[1]
        logits=logits[0]
        labels=labels.squeeze(1)
        labels=torch.cuda.LongTensor(labels.cpu().numpy())
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        loss2=self.criteria(aux_logits,labels).view(-1)
        loss2,_=torch.sort(loss2,descending=True)
        loss_mean=torch.mean(loss)
        loss2_mean=torch.mean(loss2)
        scale=1/(torch.abs(loss2_mean-loss_mean)/loss_mean)
        return loss_mean+scale*loss2_mean

