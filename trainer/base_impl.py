import os
import shutil
from abc import ABC
from typing import Callable

import torch
from PIL import ImageFile
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from torchvision import datasets

import network
from base import BaseTrainer
from utils.util import get_time
from loss.loss_builder import build_loss

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImplBaseTrainer(BaseTrainer, ABC):
    def __init__(self):
        self.args = None
        self.model_ = None
        self.optimizer_ = None
        self.train_loader_ = None
        self.val_loader_ = None
        self.lr_scheduler_ = None
        self.criterion = None
        self.writer = None
        self.losses = None

    def create_model(self, model_names: str, pretrained: bool, **kwargs):
        model = network.__dict__[model_names](pretrained=pretrained, **kwargs)
        return model

    def resume(self, model_path: str, strict: bool = False, **kwargs) -> None:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            static_dict = checkpoint['state_dict']
            self.model_.load_state_dict(static_dict, strict=strict)
            print("{} 权重resume成功。".format(model_path))
        else:
            print("{} 权重文件不存在。".format(model_path))

    def save_checkpoint(self, state, save_path: str, **kwargs) -> None:

        save_str = "checkpoint.pth.tar"

        losses = kwargs.get('losses')
        if losses is not None:
            save_str = f'loss:{losses}_' + save_str  # e.g.loss:0.001_checkpoint.pth.tar

        curr_time = get_time()
        save_str = f'{curr_time}_' + save_str

        filename = os.path.join(save_path, save_str)  # e.g.2022-11-14-(17:35:42)_checkpoint.pth.tar

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("create path {} to save checkpoint.".format(save_path))

        torch.save(state, filename)

        is_best = kwargs.get('is_best')
        if is_best is not None:
            # e.g.2022-11-14-(17:35:42)_model_best.pth.tar
            shutil.copyfile(filename, os.path.join(save_path, f"{curr_time}_model_best.pth.tar"))

    def define_scalar(self, save_path: str, comment: str, **kwargs):
        curr_time = get_time()
        logdir = os.path.join(save_path, curr_time + comment)
        writer = SummaryWriter(logdir, write_to_disk=False)
        return writer

    def define_optimizer(self, lr: float, **kwargs):
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=lr)
        return optimizer

    def define_loader(self, path: str, **kwargs):
        trans = kwargs.get('transform')
        dataset_img = datasets.ImageFolder(path, transform=trans)
        data_loader = DataLoader(dataset_img, **kwargs)
        return data_loader

    def define_criterion(self, criterion_list: list, gpus: int, **kwargs):
        criterion = []
        for criterion_name in criterion_list:
            criterion.append(build_loss(criterion_name, **kwargs).cuda(gpus))
        return criterion

    def define_lr_scheduler(self, optimizer: Callable, **kwargs):
        lrs = OneCycleLR(optimizer, **kwargs)
        return lrs
