from typing import Callable

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.optim.lr_scheduler import OneCycleLR

from dataset.data_pipe import get_loader_multi_label_class
from utils.meter import AverageMeter
from utils.util import accuracymultilabel
from base_impl import ImplBaseTrainer


class MultiLabelClassifyTrainer(ImplBaseTrainer):
    def __init__(self):
        super(MultiLabelClassifyTrainer, self).__init__()

    def init_trainer(self,
                     args,
                     train_transform: Callable,
                     tensor_transform: Callable,
                     tensorboard_save_path: str,
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

        self.lr_scheduler_ = self.define_lr_scheduler(self.optimizer_,
                                                      max_lr=self.args.lr,
                                                      steps_per_epoch=len(self.train_loader_),
                                                      epochs=self.args.epochs,
                                                      pct_start=0.2
                                                      )
        self.criterion = self.define_criterion(self.args.criterion_list, self.args.gpus, **kwargs)
        self.writer = self.define_scalar(tensorboard_save_path, comment=self.optimizer_.__module__)
        # self.judge_cycle_train = judge_cycle_train

    def fit(self, save_path: str, **kwargs):
        best_acc = 0.5
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.train_epoch(epoch, **kwargs)
            top1_acc = self.validate(epoch)
            is_best = top1_acc >= best_acc
            best_acc = max(top1_acc, best_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model_.state_dict(),
                    "best_acc1": top1_acc,
                },
                save_path,
                is_best=is_best,
                losses=self.losses
            )

    def train_epoch(self, epoch: int, **kwargs):
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
                print(e, "请检查config配置的classes是不是实际训练集的类别数。")

            acc1, accmulti = accuracymultilabel(output, target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            topmulti.update(accmulti[0], images.size(0))

            self.losses = round(float(losses.avg), 6)

            if i % self.args.print_freq == 0:
                # 这里进行记录，是为了避免过多的数据跳动。
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

    def validate(self, epoch: int, **kwargs):
        """
        此函数只验证分类准确度、分类的损失。其也可以用于多任务的类别分支的准确度预测。
        Args:
            top_k: 进行验证的类别top几分数
            args: 进行预测的参数配置
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
