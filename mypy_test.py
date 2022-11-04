def add(a,b):
    return  a+b

def accuracy(pred,
             target,
             topk,
             thresh =None):
    assert isinstance(topk, (int, tuple, list))

    if isinstance(topk, int):
        topk = (topk,)

    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu

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
    return res