import os.path

import torch
from datetime import datetime
from typing import Callable, List, Any
from loguru import logger
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from augment.aug_transforms import AugTransform
from augment.transforms_keypoints import AugKeypoints
from augment.aug_config import config_aug
from utils.util import load_aug_config

from trainer.base_trainer import BaseTrainer
from trainer.segmentation_trainer import SegmantationTrainer
from trainer.embedding_trainer import EmbeddingTrainer
from trainer.multi_label_classify_trainer import MultiLabelClassifyTrainer
from trainer.multi_task_trainer import MultiTaskTrainer

# from lr_scheduler.lr_adjustment import
_Network_ = {
    'basetrainer': BaseTrainer,
    'segmentation': SegmantationTrainer,
    'embedding_classification': EmbeddingTrainer,
    'multilabel_classification': MultiLabelClassifyTrainer,
    'multitask': MultiTaskTrainer,
}

__all__ = [
    'build_optimizer',
    'build_lr_scheduler',
    'build_loss',
    'build_model',
    'build_dataloader',
    'build_summarywriter'
]


def build_optimizer(optim: Callable, **kwargs) -> Callable:
    optimizer = optim(**kwargs)
    return optimizer


def build_lr_scheduler(scheduler: Callable, **kwargs) -> Callable:
    lr_scheduler = scheduler(**kwargs)
    return lr_scheduler


def build_model(network_type: str, **kwargs):
    if network_type not in _Network_.keys():
        logger.error(f'Do not find {network_type} network -> exit')

    trainer = _Network_[network_type]
    trainer.init_trainer(**kwargs)


def build_loss(loss_name: List[str, ...], **kwargs) -> list:
    losses = []
    for loss in loss_name:
        losses.append(eval(loss)(**kwargs))
    return losses


def build_dataloader(path: str, transform: Callable, **kwargs):
    dataset = datasets.ImageFolder(path, transform=transform)
    data_loader = DataLoader(dataset, **kwargs)
    return data_loader


def build_summarywriter(save_path: str, comment: str = '', **kwargs):
    curr_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(save_path, curr_time + '_' + comment)
    writer = SummaryWriter(log_dir, **kwargs)
    return writer


if __name__ == '__main__':
    ...
