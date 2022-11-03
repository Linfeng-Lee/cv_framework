import torch
import numpy as np


def accuracy(pred: torch.Tensor, target: torch.Tensor, topk: int = 1, thresh: float = None):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu

    maxk = max(topk)
    assert pred.ndim == 2 and target.ndim == 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), f'maxk{maxk} exceeds pred dimension {pred.size(1)}'

    pred_val, pred_idx = pred.topk(maxk, dim=1)
    pred_idx = pred_idx.t()  # change shape:(bs,nc)->(nc,bs)
    # target.shape:(bs,0)->(1,bs)->(nc,bs)
    correct = pred_idx.eq(target.view(1, -1).expand_as(pred_idx))
    if thresh is not None:
        correct = correct & (pred_val > thresh).t()
    res = []
    bs = pred.size(0)
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / bs))
    return res[0] if return_single else res


def top1(pred: torch.Tensor, target: torch.Tensor, thresh: float = None) -> torch.Tensor:
    return accuracy(pred, target, 1, thresh)


def top5(pred: torch.Tensor, target: torch.Tensor, thresh: float = None) -> list:
    return accuracy(pred, target, 5, thresh)


# n:int or tuple
def topn(pred: torch.Tensor, target: torch.Tensor, n, thresh: float = None) -> list:
    return accuracy(pred, target, n, thresh)


def accuracymultilabel(output: torch.Tensor, target: torch.Tensor):
    with torch.no_grad():
        num_instance, num_class = target.size()
        output[output > 0.5] = 1
        output[output <= 0.5] = 0

        count = 0
        gbcount = 0
        for i in range(num_instance):
            p = sum(np.logical_and(target[i].cpu().detach().numpy(), output[i].cpu().detach().numpy()))
            q = sum(np.logical_or(target[i].cpu().detach().numpy(), output[i].cpu().detach().numpy()))
            count += p / q

            if output[i][0] == target[i][0]:
                gbcount += 1

    return [gbcount / num_instance], [count / num_instance]


def accuracy_good_ng(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred_copy = pred.clone()
        pred_copy[pred_copy > 0] = 1
        target_copy = target.clone()
        target_copy[target_copy > 0] = 1
        correct = pred_copy.eq(target_copy.view(1, -1).expand_as(pred_copy))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
