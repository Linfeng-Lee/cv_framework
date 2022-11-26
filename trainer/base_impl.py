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
        self.scaler = None

    def create_model(self, model_names: str, pretrained: bool, **kwargs):
        model = network.__dict__[model_names](pretrained=pretrained, **kwargs)
        return model

    def resume(self, model_path: str, strict: bool = False, map_location: str = 'cpu', **kwargs) -> None:

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=map_location)

            self.model_.load_state_dict(checkpoint.get('state_dict'), strict=strict)
            self.lr_scheduler_.load_state_dict(checkpoint.get('lr_scheduler'))
            self.optimizer_.load_state_dict(checkpoint.get('optimizer'))
            self.args.start_epoch = checkpoint.get('epoch') + 1

            if self.args.amp:
                self.scaler.load_state_dict(checkpoint.get('scaler'))

            print("{} 权重resume成功。".format(model_path))
        else:
            print("{} 权重文件不存在。".format(model_path))

    def save_checkpoint(self, state_dict: dict, save_path: str, **kwargs) -> None:

        save_str = "checkpoint.pth"

        curr_time = get_time()
        save_str = f'{curr_time}_' + save_str

        filename = os.path.join(save_path, save_str)  # e.g.2022-11-14-(17:35:42)_checkpoint.pth.tar

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("create path {} to save checkpoint.".format(save_path))

        save_dict = {
            "epoch": kwargs.get('epoch'),
            "state_dict": state_dict,
            'lr_scheduler': kwargs.get('lr_scheduler'),
            'optimizer': kwargs.get('optimizer'),
            "best": kwargs.get('best'),
        }
        if self.args.amp:
            save_dict['scaler'] = kwargs.get('amp'),

        torch.save(save_dict, filename)

        is_best = kwargs.get('is_best')
        if is_best is not None:
            # e.g.2022-11-14-(17:35:42)_model_best.pth.tar
            shutil.copyfile(filename, os.path.join(save_path, f"{curr_time}_model_best.pth"))

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
