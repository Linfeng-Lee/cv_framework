import time
import sys
from typing import Callable
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from loguru import logger
from trainer.base_trainer import BaseTrainer
from dataset.data_pipe import get_loader_multi_label_class

from utils.meter import AverageMeter
from utils.meter import ProgressMeter
from utils.util import accuracymultilabel
import os
from PIL import Image
import shutil
from torch.optim.lr_scheduler import OneCycleLR


class MultiLabelClassifyTrainer(BaseTrainer):
    def __init__(self):
        super(MultiLabelClassifyTrainer, self).__init__()
        # self.train_loader_remove_no_json_ = None

    def init_trainer_v2(self,
                        args,
                        train_transform: Callable,
                        tensor_transform: Callable,
                        tensorboard_save_path: str = 'temp/tensorboard',
                        **kwargs
                        ):
        self.args = args

        # self.args = kwargs["args"]
        self.model_ = self.create_model(self.args.net,
                                        self.args.pretrained,
                                        num_classes=self.args.classes).cuda(self.args.gpus)
        self.optimizer_ = self.define_optimizer(self.args.lr)
        self.train_loader_ = get_loader_multi_label_class(self.args.train_data_path,
                                                          train_transform,
                                                          self.args.balance_n_classes,
                                                          self.args.balance_n_samples,
                                                          self.args.class_id_map_save_path,
                                                          tensor_transform, )

        self.val_loader_ = get_loader_multi_label_class(self.args.val_data_path,
                                                        tensor_transform,
                                                        self.args.balance_n_classes,
                                                        self.args.balance_n_samples,
                                                        self.args.class_id_map_save_path,
                                                        tensor_transform,
                                                        is_training=False)

        self.lr_scheduler_ = self.define_lr_scheduler(self.optimizer_)
        self.criterion = self.define_criterion(self.args.criterion_list, self.args.gpus, **kwargs)
        self.writer = self.define_scalar(tensorboard_save_path, comment=self.optimizer_.__module__)
        # self.judge_cycle_train = judge_cycle_train

    def init_trainer(self,
                     model_names: str,
                     lr: float,
                     train_data_path: str,
                     val_data_path: str,
                     gpus: int,
                     classes: int,
                     classify_transform,
                     criterion_list: list = None,
                     tensorboard_save_path: str = 'temp/tensorboard',
                     pretrained: bool = False,
                     judge_cycle_train: bool = False,
                     **kwargs):
        self.epochs = kwargs["epoches"]
        self.args = kwargs["args"]
        self.lr = lr
        self.model_ = self.create_model(model_names, pretrained, num_classes=classes).cuda(gpus)
        self.optimizer_ = self.define_optimizer(lr)
        self.classify_transform = classify_transform
        self.train_loader_ = get_loader_multi_label_class(train_data_path,
                                                          classify_transform,
                                                          kwargs["balance_n_classes"],
                                                          kwargs["balance_n_samples"],
                                                          kwargs["class_id_map_save_path"],
                                                          kwargs["tensor_transform"],
                                                          kwargs["weights"])
        self.val_loader_ = get_loader_multi_label_class(val_data_path,
                                                        kwargs["tensor_transform"],
                                                        kwargs["balance_n_classes"],
                                                        kwargs["balance_n_samples"],
                                                        kwargs["class_id_map_save_path"],
                                                        kwargs["tensor_transform"],
                                                        kwargs["weights"],
                                                        is_training=False)
        self.lr_scheduler_ = self.define_lr_scheduler(self.optimizer_)
        self.criterion = self.define_criterion(criterion_list, gpus, **kwargs)
        self.writer = self.define_scalar(tensorboard_save_path, comment=self.optimizer_.__module__)
        self.judge_cycle_train = judge_cycle_train

    def fit(self,
            save_path: str,
            **kwargs):
        best_acc = 0.5
        for epoch in range(self.args.start_epoch, self.args.epochs):
            # self.lr_scheduler_.step()
            self.train_epoch(epoch, **kwargs)
            top1_acc = self.validate(epoch)
            is_best = top1_acc >= best_acc
            best_acc = max(top1_acc, best_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model_.state_dict(),
                    "best_acc1": top1_acc,
                }, is_best, save_path
            )

    def define_lr_scheduler(self, optimizer):
        lrs = OneCycleLR(optimizer,
                         max_lr=self.args.lr,
                         steps_per_epoch=len(self.train_loader_),
                         epochs=self.args.epochs,
                         pct_start=0.2)

        return lrs

    def train_epoch(self,
                    epoch: int,
                    **kwargs):
        losses = AverageMeter('Loss_class', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        topmulti = AverageMeter("accmulti", ':6.2f')

        self.model_.train()
        train_len = len(self.train_loader_)
        start_iter = epoch * train_len
        self.writer.add_scalar("lr", self.optimizer_.param_groups[0]['lr'], epoch)

        pbar = tqdm(enumerate(self.train_loader_),
                    total=train_len,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for i, (images, target) in pbar:

            if torch.cuda.is_available():
                images = images.cuda(self.args.gpus)
                target = target.cuda(self.args.gpus, non_blocking=True)

            output = self.model_(images)

            loss = self.criterion[0](output, target)

            try:
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()
                self.lr_scheduler_.step()

            except Exception as e:
                print(e, "?????????config?????????classes???????????????????????????????????????")

            acc1, accmulti = accuracymultilabel(output, target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            topmulti.update(accmulti[0], images.size(0))

            if i % self.args.print_freq == 0:
                # ????????????????????????????????????????????????????????????
                self.writer.add_scalar("train_loss", losses.avg, start_iter + i)
                self.writer.add_scalar("train_acc1", top1.avg, start_iter + i)

                _top1 = round(float(acc1[0]), 4)
                _topmulti = round(float(topmulti.avg), 4)
                _lr = round(float(self.optimizer_.param_groups[0]['lr']), 6)
                _loss = round(float(losses.avg), 6)

                logger.debug(f'\n[Training] | '
                             f'[{epoch}/{self.args.epochs}] | '
                             f'{losses.name} : {_loss} | '
                             f'lr : {_lr} | '
                             f'{top1.name} : {_top1} | '
                             f'{topmulti.name} : {_topmulti}')

    def validate(self,
                 epoch: int,
                 **kwargs):
        """
        ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        Args:
            top_k: ?????????????????????top?????????
            args: ???????????????????????????
        Returns:
        """

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        topmulti = AverageMeter("accmulti", ':6.2f')

        confusion_matrix_multilabel = np.zeros((2 * self.args.classes, 2 * self.args.classes), dtype=np.long)

        self.model_.eval()

        with torch.no_grad():
            for i, sample in enumerate(self.val_loader_):
                if len(sample) == 2:
                    images, target = sample
                else:
                    images, target = sample[:2]

                if torch.cuda.is_available():
                    images = images.cuda(self.args.gpus)
                    target = target.cuda(self.args.gpus, non_blocking=True)

                output = self.model_(images)
                if isinstance(output, list):
                    output = output[0]

                loss = self.criterion[0](output, target)

                acc1, accmulti = accuracymultilabel(output, target)
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                topmulti.update(accmulti[0], images.size(0))

                output[output > 0.5] = 1
                output[output <= 0.5] = 0

                for t, p in zip(target.long(), output.long()):
                    for j, (t0, p0) in enumerate(zip(t, p)):
                        confusion_matrix_multilabel[2 * j + 1 - t0, 2 * j + 1 - p0] += 1

                if i % self.args.print_freq == 0:
                    _top1 = round(float(top1.avg), 4)
                    _topmulti = round(float(topmulti.avg), 4)
                    _lr = round(float(self.optimizer_.param_groups[0]['lr']), 6)
                    _loss = round(float(losses.avg), 6)

                    logger.debug(f'\n[Val] | '
                                 f'[{epoch}/{self.args.epochs}] | '
                                 f'{losses.name} : {_loss} | '
                                 f'lr : {_lr} | '
                                 f'{top1.name} : {_top1} | '
                                 f'{topmulti.name} : {_topmulti}')

        logger.info(f'confusion_matrix_multilabel:{confusion_matrix_multilabel}')

        self.writer.add_scalar("val_acc1", top1.avg, epoch)

        return top1.avg

    @torch.no_grad()
    def check_data_conflict(self, args, transforms, tag="train"):
        print("check {} config conflict...".format(tag))
        if tag not in ["train", "test"]:
            raise ValueError("tag either be 'train' or 'test'!")
        self.model_.eval()
        label_map = sorted(os.listdir(os.path.join(args.data, tag)))
        image_paths = self._get_images(os.path.join(args.data, tag))
        for p in image_paths:
            _ = p.split(os.sep)
            gt = _[_.index(tag) + 1]
            gt_id = label_map.index(gt)
            img = Image.open(p).convert('RGB')

            img_pil = img.resize((args.input_w, args.input_h), Image.BILINEAR)
            img = transforms(img_pil)
            input = img.unsqueeze(0).cuda(args.gpus)
            predict = self.model_(input)

            pred_gs = predict.sigmoid().squeeze()[0].item()
            pred_cls_id = 0 if pred_gs > 0.5 else 1

            pred_score = predict.sigmoid().squeeze().cpu().detach().numpy()
            pred_idx = (pred_score > 0.5)

            # conflict type 1: good score conflict
            if (gt_id > 0 and pred_cls_id == 0) or (gt_id == 0 and pred_cls_id > 0):
                # ng predicted as ok || ok predicted as ng
                dst_path = os.path.join(args.data, "conflict-" + tag, gt)
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                shutil.move(p, dst_path)
                shutil.move(p.replace(".png", ".json"), dst_path)

            # conflict type2: good label come up with ng label
            elif pred_idx[0] and pred_idx[1:].any():
                dst_path = os.path.join(args.data, "conflict-" + tag, gt)
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                shutil.move(p, dst_path)
                shutil.move(p.replace(".png", ".json"), dst_path)

                # conflict type3: no label
            elif not pred_idx.any():
                dst_path = os.path.join(args.data, "conflict-" + tag, gt)
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                shutil.move(p, dst_path)
                shutil.move(p.replace(".png", ".json"), dst_path)

        print("done")

    # def define_lr_scheduler(self, optimizer):
    #     steps_per_epoch = len(self.train_loader_)
    #     lr_scheduler = OneCycleLR(optimizer,
    #                               max_lr=self.lr,
    #                               steps_per_epoch=steps_per_epoch,
    #                               epochs=self.epochs,
    #                               pct_start=0.2)
    #     return lr_scheduler
