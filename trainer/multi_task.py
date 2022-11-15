import time
from abc import ABC
from typing import Callable

import torch
import numpy as np
from torch import nn
import torch.cuda.amp as amp

from dataset.data_pipe import get_loader_mask, get_loader_class
from lr_scheduler.lr_adjustment import LambdaLR
from utils.meter import AverageMeter
from utils.meter import ProgressMeter
from utils.util import accuracy
from utils.util import generate_matrix, accl_miou
from base_impl import ImplBaseTrainer


class MultiTaskTrainer(ImplBaseTrainer, ABC):
    def __init__(self):
        super(MultiTaskTrainer, self).__init__()
        self.judge_cycle_train = None
        self.val_loader_remove_no_json_ = None
        self.train_loader_remove_no_json_ = None

    def init_trainer(self,
                     args,
                     classify_transform: Callable,
                     tensor_transform: Callable,
                     val_transform: Callable,
                     tensorboard_save_path='temp/tensorboard',
                     **kwargs):
        self.args = args
        self.model_ = self.create_model(self.args.model_names,
                                        self.args.pretrained,
                                        num_classes=self.args.classes,
                                        **kwargs).cuda(self.args.gpus)
        self.optimizer_ = self.define_optimizer(self.args.lr)
        self.train_loader_ = get_loader_class(self.args.train_data_path,
                                              classify_transform,
                                              self.args.balance_n_classes,
                                              self.args.balance_n_samples,
                                              self.args.class_id_map_save_path)
        # self.train_loader_remove_no_json_= self.define_loader(train_data_path, **kwargs)
        self.train_loader_remove_no_json_ = self.define_loader(self.args.train_data_path,
                                                               tensor_transform=tensor_transform,
                                                               is_training=True,
                                                               **kwargs)
        # self.val_loader_ = self.define_loader(val_data_path, **kwargs)
        self.val_loader_remove_no_json_ = self.define_loader(self.args.val_data_path,
                                                             tensor_transform=val_transform,
                                                             is_training=False,
                                                             **kwargs)
        self.lr_scheduler_ = self.define_lr_scheduler(self.optimizer_)
        self.criterion = self.define_criterion(self.args.criterion_list,
                                               self.args.gpus,
                                               **kwargs)
        self.writer = self.define_scalar(tensorboard_save_path, comment=self.optimizer_.__module__)
        self.judge_cycle_train = self.args.judge_cycle_train

    def define_lr_scheduler(self, optimizer: Callable, **kwargs):
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch / 4 + 1))
        return lr_scheduler

    def define_loader(self, path: str, **kwargs):
        data_loader = get_loader_mask(path, **kwargs)
        return data_loader

    def validate(self, epoch: int, **kwargs):
        """
        Args:
            epoch: current epoch of validate
        Returns:
        """
        batch_time = AverageMeter('Time', ':6.3f')
        loss_cls_output_avg = AverageMeter('loss_mcls', ':.4e')
        loss_mask_output_avg = AverageMeter('loss_mask', ':.4e')

        top1 = AverageMeter('Acc@1', ':6.2f')
        # mIOU = AverageMeter('mIOU', ':.4e')
        progress = ProgressMeter(
            len(self.val_loader_remove_no_json_),
            [batch_time, loss_cls_output_avg, loss_mask_output_avg, top1],
            prefix="Epoch: [{}]".format(epoch))

        self.model_.eval()
        with torch.no_grad():
            end = time.time()
            i = 0

            confusion_matrix = np.zeros((self.args.classes, self.args.classes))
            # output2
            confusion_matrix1 = np.zeros((self.args.classes, self.args.classes))

            for no_json_list in self.val_loader_remove_no_json_:
                # 为了兼容后期返回的各种数量长度
                if isinstance(no_json_list, list):
                    images_remove_no_json, target_remove_no_json, mask_target_remove_no_json = no_json_list[:3]
                else:
                    raise RuntimeError("iter is not list!")

                # 进行加mask分支训练
                if torch.cuda.is_available():
                    images_remove_no_json = images_remove_no_json.cuda(self.args.gpus)
                    target_remove_no_json = target_remove_no_json.cuda(self.args.gpus, non_blocking=True)
                    mask_target_remove_no_json = mask_target_remove_no_json.cuda(self.args.gpus)
                for cur_time in range(self.args.cycle_train_times):
                    cls_output, mask_output = self.model_(images_remove_no_json)  #
                    # loss_cls = self.criterion[0](cls_output, target_remove_no_json)
                    loss_cls = 0
                    if isinstance(cls_output, list):
                        for n_l in range(len(cls_output)):
                            loss_cls += self.criterion[0](cls_output[n_l], target_remove_no_json)
                        loss_cls = loss_cls / len(cls_output)
                    else:
                        loss_cls = self.criterion[0](cls_output, target_remove_no_json)
                    loss_mask = self.criterion[1]([mask_output[0], mask_output[2]], mask_target_remove_no_json)
                    loss_mask += 0.5 * self.criterion[1]([mask_output[1], mask_output[3]], mask_target_remove_no_json)

                    if isinstance(cls_output, list):
                        acc1, acc5 = accuracy(cls_output[0], target_remove_no_json, topk=(1, self.args.top_k))
                    else:
                        acc1, acc5 = accuracy(cls_output, target_remove_no_json, topk=(1, self.args.top_k))

                    confusion_matrix += generate_matrix(self.args.classes, mask_output[0], mask_target_remove_no_json)

                    # output2
                    confusion_matrix1 += generate_matrix(self.args.classes, mask_output[1], mask_target_remove_no_json)

                loss_cls_output_avg.update(loss_cls.item(), images_remove_no_json.size(0))
                loss_mask_output_avg.update(loss_mask.item(), images_remove_no_json.size(0))

                top1.update(acc1[0], images_remove_no_json.size(0))
                # mIOU.update(miou, images_remove_no_json.size(0))
                batch_time.update(time.time() - end)

                end = time.time()

                if i % self.args.print_freq == 0:
                    progress.display(i)

                i += 1

            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

            # TODO: 返回miou
            iou, miou = accl_miou(confusion_matrix)
            print(' * iou {iou}, mIOU {miou:.3f} '.format(iou=iou, miou=miou))

            # output2
            iou1, miou1 = accl_miou(confusion_matrix1)
            print(' * iou1 {iou1}, mIOU1 {miou1:.3f} '.format(iou1=iou1, miou1=miou1))

        self.writer.add_scalar("val_acc1", top1.avg, epoch)
        self.writer.add_scalar("miou", miou, epoch)
        return [miou, top1.avg]

    def fit(self, save_path: str, **kwargs):
        best = 0.0
        epochs_best = 0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.lr_scheduler_.step()
            self.train_epoch(epoch, **kwargs)
            top1 = self.validate(epoch)
            if isinstance(top1, list):
                if not isinstance(best, list):
                    best = [best, 0.0]
                is_best = top1[0] >= best[0]
                if top1[0] >= best[0]:
                    best = top1
                    epochs_best = epoch
            else:
                is_best = top1 >= best
                if top1 >= best:
                    best = top1
                    epochs_best = epoch

            # best=max(top1,best)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model_.state_dict(),
                    "best": top1,
                },
                save_path=save_path,
                is_best=is_best
            )
            if isinstance(best, list):
                print("best: {best1:.3f}, best: {best2:.3f}%".format(best1=best[0], best2=best[1]))
                print("epoch: {epochs}".format(epochs=epochs_best))
            else:
                print("best: {best:.3f}, epoch: {epochs}".format(best=best, epochs=epochs_best))

    def train_epoch(self, epoch: int, **kwargs):
        topn_str = "Acc@{}".format(self.args.top_k)
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss_class', ':.4e')
        loss_mask_output_avg = AverageMeter('loss_mask', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        topn = AverageMeter(topn_str, ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader_),
            [batch_time, data_time, losses, loss_mask_output_avg, top1, topn],
            prefix="Epoch: [{}]".format(epoch))

        scaler = amp.GradScaler()
        self.model_.train()
        start_iter = epoch * len(self.train_loader_)
        end = time.time()
        self.writer.add_scalar("lr", self.optimizer_.param_groups[0]['lr'], epoch)
        # for i, (images, target) in enumerate(self.train_loader_):
        i = 0
        for (images, target), no_json_list in \
                zip(self.train_loader_, self.train_loader_remove_no_json_):
            # 为了兼容后期返回的各种数量长度
            if isinstance(no_json_list, list):
                images_remove_no_json, target_remove_no_json, mask_target_remove_no_json = no_json_list[:3]
            else:
                raise RuntimeError("iter is not list!")

            data_time.update(time.time() - end)
            if torch.cuda.is_available():
                images = images.cuda(self.args.gpus)
                target = target.cuda(self.args.gpus, non_blocking=True)
            with amp.autocast():
                output, _ = self.model_(images)  #
                # loss = self.criterion[0](output, target)
                loss = 0
                if isinstance(output, list):
                    for n_l in range(len(output)):
                        loss += self.criterion[0](output[n_l], target)
                    loss = loss / len(output)
                else:
                    loss = self.criterion[0](output, target)

            try:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer_)
                scaler.update()
                self.optimizer_.zero_grad()
            except Exception as e:
                print(e, "请检查config配置的classes是不是实际训练集的类别数。")

            if isinstance(output, list):
                acc1, acc5 = accuracy(output[0], target, topk=(1, self.args.top_k))
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, self.args.top_k))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            topn.update(acc5[0], images.size(0))
            self.writer.add_scalar("train_loss", losses.avg, start_iter + i)
            self.writer.add_scalar("train_acc1", top1.avg, start_iter + i)

            # 进行加mask分支训练
            if torch.cuda.is_available():
                images_remove_no_json = images_remove_no_json.cuda(self.args.gpus)
                target_remove_no_json = target_remove_no_json.cuda(self.args.gpus, non_blocking=True)
                mask_target_remove_no_json = mask_target_remove_no_json.cuda(self.args.gpus)

            for cur_time in range(self.args.cycle_train_times):
                with amp.autocast():
                    output, mask_output = self.model_(images_remove_no_json)  #
                    loss_cls = 0
                    if isinstance(output, list):
                        for n_l in range(len(output)):
                            loss_cls += self.criterion[0](output[n_l], target_remove_no_json)
                        loss_cls = loss_cls / len(output)
                    else:
                        loss_cls = self.criterion[0](output, target_remove_no_json)

                    loss_mask = self.criterion[1]([mask_output[0], mask_output[2]], mask_target_remove_no_json)
                    loss_mask += 0.5 * self.criterion[1]([mask_output[1], mask_output[3]], mask_target_remove_no_json)

                    loss = loss_cls + loss_mask
                    if cur_time > 0:
                        loss = loss * 0.1

                self.optimizer_.zero_grad()
                # loss.backward()
                # self.optimizer_.step()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer_)
                scaler.update()
                # self.optimizer_.zero_grad()
                # 是否关闭这个判断，有些网络构建是不适合这个判断的，如果输出的通道数是大于1的需要关闭这个功能。
                # if self.judge_cycle_train:
                #     # 进行困难样本的重新训练
                #     if not self.judge_cycle_mask_train(mask_output,mask_target_remove_no_json,args.area_threshold):
                #         break

            loss_mask_output_avg.update(loss_mask.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                progress.display(i)
                self.writer.add_scalar("train_mask_loss", loss_mask_output_avg.avg, start_iter + i)

                # self.save_checkpoint(
                #     {
                #         "epoch": epoch + 1,
                #         "state_dict": self.model_.state_dict(),
                #         "best_acc1": top1,
                #     },
                #     True
                # )

            i += 1
