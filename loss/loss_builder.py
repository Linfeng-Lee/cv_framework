
# from torch.nn import CrossEntropyLoss
from loss.crossentropy_loss import CrossEntropyLoss
from loss.iou_ssim_loss_class import IouSsimLoss
from loss.focal_loss_class import FocalLoss
from loss.dice_loss import DiceLoss
from loss.period_loss import PeriodLoss
from loss.multi_label_class_loss import BCEWithLogitsLoss,SoftmaxCrossentropyLoss
from loss.multi_label_class_loss import AsymmetricLoss

SUPPORT_LOSS={"CrossEntropyLoss",
              "FocalLoss",
              "IouSsimLoss",
              "DiceLoss",
              "PeriodLoss",
              "BCEWithLogitsLoss",
              "AsymmetricLoss"
              }

def build_loss(loss_name,**kwargs):
    assert loss_name in SUPPORT_LOSS, f"all support loss is {SUPPORT_LOSS}"
    print(loss_name)
    criterion=eval(loss_name)(**kwargs)
    return criterion
