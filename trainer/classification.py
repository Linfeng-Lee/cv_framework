from typing import Callable

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from PIL import ImageFile
from torch.optim.lr_scheduler import OneCycleLR

from utils.evaluation import topk
from utils.meter import AverageMeter
from base_impl import ImplBaseTrainer

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationTrainer(ImplBaseTrainer):
    def __init__(self):
        super(ClassificationTrainer).__init__()
        self.losses = None

    def init_trainer(self,
                     args,
                     train_transforms: Callable,
                     val_transforms: Callable,
                     tensorboard_save_path: str = 'temp/tensorboard',
                     **kwargs):
        self.args = args
        self.model_ = self.create_model(self.args.net,
                                        self.args.pretrained,
                                        num_classes=self.args.classes,
                                        **kwargs).cuda(self.args.gpus)
        self.optimizer_ = self.define_optimizer(self.args.lr)
        self.train_loader_ = self.define_loader(self.args.train_data_path,
                                                transform=train_transforms,
                                                batch_size=self.args.batch_size,
                                                num_workers=self.args.workers,
                                                shuffle=True)
        self.val_loader_ = self.define_loader(self.args.val_data_path,
                                              transform=val_transforms,
                                              batch_size=self.args.batch_size,
                                              num_workers=self.args.workers,
                                              shuffle=True)
        self.lr_scheduler_ = self.define_lr_scheduler(self.optimizer_,
                                                      max_lr=self.args.lr,
                                                      steps_per_epoch=len(self.train_loader_),
                                                      epochs=self.args.epochs,
                                                      pct_start=0.2)
        self.criterion = self.define_criterion(self.args.criterion_list, self.args.gpus)
        self.writer = self.define_scalar(tensorboard_save_path, comment=self.optimizer_.__module__)

        if isinstance(self.args.topk, (int, list, tuple)) is False:
            logger.error(f'args.topk type error,only support:(int,list, tuple) -> exit.')
            exit()

    def fit(self, save_path: str, **kwargs):
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
                },
                save_path,
                is_best=is_best,
                losses=self.losses
            )

    def train_epoch(self, epoch: int, **kwargs):
        losses = AverageMeter('Loss', ':.4e')
        if isinstance(self.args.topk, (list, tuple)):
            topn = [AverageMeter(f'Acc@{k}') for k in self.args.topk]
        else:
            topn = [AverageMeter(f'Acc@{self.args.topk}')]

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

            acc = topk(output, target, k=self.args.topk)

            losses.update(loss.item(), images.size(0))
            self.losses = round(float(losses.avg), 6)
            for j in range(len(self.args.topk)):
                topn[j].update(acc[j], images.size(0))

            self.optimizer_.zero_grad()
            loss.backward()
            self.optimizer_.step()
            self.lr_scheduler_.step()

            if i % self.args.print_freq == 0:
                # 这里进行记录，是为了避免过多的数据跳动。
                normalize_transpose = kwargs.get('normalize_transpose')
                origin_input = normalize_transpose(images)
                origin_input = np.array(origin_input.cpu().detach().numpy(), dtype=np.uint8)
                self.writer.add_images('input_img', origin_input, (start_iter + i))
                self.writer.add_scalar("train_loss", losses.avg, (start_iter + i))
                for k in range(len(self.args.topk)):
                    self.writer.add_scalar(f"train_acc{k}", topn[k].avg, (start_iter + i))

                _lr = round(float(self.optimizer_.param_groups[0]['lr']), 6)
                _loss = round(float(losses.avg), 6)
                topn_str = ''
                for k in topn:
                    topn_str += f' | {k.name}:{round(float(k.avg), 4)}'

                logger.debug(f'[Training] | '
                             f'[{epoch}/{self.args.epochs}] | '
                             f'Loss:{_loss} | '
                             f'lr:{_lr} | ' +
                             topn_str)

    def validate(self, epoch: int, **kwargs):
        """
        Args:
            epoch: current epoch
        Returns: top1.data
        """
        losses = AverageMeter('Loss')

        if isinstance(self.args.topk, (list, tuple)):
            topn = [AverageMeter(f'Acc@{k}') for k in self.args.topk]
        else:
            topn = [AverageMeter(f'Acc@{self.args.topk}')]

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

                # update losses
                loss = self.criterion[0](output, target)
                losses.update(loss.item(), images.size(0))

                # get the topn
                acc = topk(output, target, k=self.args.topk)
                for j in range(len(self.args.topk)):
                    topn[j].update(acc[j], images.size(0))

                # display information
                topn_str = ''
                for n in topn:
                    topn_str += f' | {n.name}:{round(float(n.avg), 4)}'

                if i % self.args.print_freq == 0:
                    logger.debug(f'[Val] | '
                                 f'{losses.name}: {round(float(losses.avg), 4)}' +
                                 topn_str)

        self.writer.add_scalar("val_acc1", topn[0].avg, epoch)

        return topn[0].avg
