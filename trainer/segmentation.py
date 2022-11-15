import os
import json
from typing import Callable

import cv2
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.optim.lr_scheduler import OneCycleLR

from dataset.data_pipe import get_loader_mask
from utils.meter import AverageMeter
from utils.meter import SegmentationEvaluator
from base_impl import ImplBaseTrainer


class SegmentationTrainer(ImplBaseTrainer):
    def __init__(self):
        super(SegmentationTrainer, self).__init__()
        self.evaluator = None
        self.train_loader_remove_no_json_ = None

    def init_trainer(self,
                     args,
                     tensor_transform: Callable,
                     mask_transform: Callable,
                     tensorboard_save_path: str,
                     **kwargs
                     ):
        # self.args = kwargs["args"]
        self.args = args
        self.model_ = self.create_model(self.args.net,
                                        self.args.pretrained,
                                        num_classes=self.args.classes).cuda(self.args.gpus)
        self.optimizer_ = self.define_optimizer(self.args.lr)
        self.train_loader_remove_no_json_ = self.define_loader(self.args.train_data_path,
                                                               tensor_transform=tensor_transform,
                                                               mask_transform=mask_transform,
                                                               balance_n_classes=self.args.balance_n_classes,
                                                               balance_n_samples=self.args.balance_n_samples,
                                                               class_id_map_save_path=self.args.class_id_map_save_path,
                                                               **kwargs
                                                               )
        self.val_loader_ = self.define_loader(self.args.val_data_path,
                                              tensor_transform=tensor_transform,
                                              mask_transform=mask_transform,
                                              balance_n_classes=self.args.balance_n_classes,
                                              balance_n_samples=self.args.balance_n_samples,
                                              class_id_map_save_path=self.args.class_id_map_save_path,
                                              **kwargs)
        self.lr_scheduler_ = self.define_lr_scheduler(self.optimizer_,
                                                      max_lr=self.args.lr,
                                                      steps_per_epoch=len(self.train_loader_remove_no_json_),
                                                      epochs=self.args.epochs,
                                                      pct_start=0.2)
        self.criterion = self.define_criterion(self.args.criterion_list, self.args.gpus, **kwargs)
        self.writer = self.define_scalar(tensorboard_save_path, comment=self.optimizer_.__module__)
        self.evaluator = SegmentationEvaluator(self.args.classes)

    def define_loader(self, path: str, **kwargs):
        data_loader = get_loader_mask(path, **kwargs)
        return data_loader

    @torch.no_grad()
    def validate(self, epoch, **kwargs):
        i = 0
        self.model_.eval()
        self.evaluator.reset()

        val_loader_len = len(self.val_loader_)
        pbar = tqdm(self.val_loader_,
                    total=val_loader_len,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for no_json_list in pbar:  # img_tensor,class_id,mask_trg,img_path
            # 为了兼容后期返回的各种数量长度
            # data_time.update(time.time() - end)

            if isinstance(no_json_list, list):
                images_remove_no_json, target_remove_no_json, mask_target_remove_no_json = no_json_list[:3]
            else:
                raise RuntimeError("iter is not list!")

            # 进行加mask分支训练
            if torch.cuda.is_available():
                images_remove_no_json = images_remove_no_json.cuda(self.args.gpus)
                mask_target_remove_no_json = mask_target_remove_no_json.cuda(self.args.gpus)

            mask_output = self.model_(images_remove_no_json)

            # calculate batch mIOU        
            pred = mask_output[0].detach().cpu().softmax(dim=1).argmax(dim=1).numpy().astype("int")
            gt = mask_target_remove_no_json.detach().cpu().squeeze(1).numpy().astype("int")
            self.evaluator.add_batch(gt, pred)
            batch_miou = self.evaluator.Mean_Intersection_over_Union()

            # debug:
            # _=[]
            # for j in range(pred.shape[0]):
            #     _.append(self._cal_iou(pred[j],gt[j],args.classes,reduction="mean"))
            # _=sum(_)/len(_)
            # print(batch_miou)
            # print(_)
            # debug

            # batch_time.update(time.time() - end)
            # end = time.time()

            if i % self.args.print_freq == 0:
                logger.debug(f'[Val] | mIoU:{batch_miou}')

            i += 1

        self.evaluator.reset()

        return 1.0

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

    def train_epoch(self,
                    epoch: int,
                    **kwargs):
        loss_mask_output_avg = AverageMeter('loss_mask', ':.4e')

        self.model_.train()
        train_len = len(self.train_loader_remove_no_json_)
        start_iter = epoch * train_len
        # end = time.time()
        _lr = self.optimizer_.param_groups[0]['lr']
        self.writer.add_scalar("lr", _lr, epoch)
        i = 0
        self.evaluator.reset()
        pbar = tqdm(self.train_loader_remove_no_json_,
                    total=train_len,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for no_json_list in pbar:  # img_tensor,class_id,mask_trg,img_path
            # data_time.update(time.time() - end)
            # 为了兼容后期返回的各种数量长度
            if isinstance(no_json_list, list):
                images_remove_no_json, target_remove_no_json, mask_target_remove_no_json = no_json_list[:3]
            else:
                raise RuntimeError("iter is not list!")

            # 进行加mask分支训练
            if torch.cuda.is_available():
                images_remove_no_json = images_remove_no_json.cuda(self.args.gpus)
                mask_target_remove_no_json = mask_target_remove_no_json.cuda(self.args.gpus)

            for cur_time in range(self.args.cycle_train_times):
                mask_output = self.model_(images_remove_no_json)
                loss_mask = self.criterion[0](mask_output, mask_target_remove_no_json)
                loss = loss_mask
                if cur_time > 0:
                    loss = loss * 0.1

                self.optimizer_.zero_grad()
                loss.backward()

                self.optimizer_.step()
                self.lr_scheduler_.step()
                # 进行困难样本的重新训练
                if not self.judge_cycle_mask_train(mask_output, target_remove_no_json, self.args.area_threshold):
                    break

            # calculate batch mIOU        
            pred = mask_output[0].detach().cpu().softmax(dim=1).argmax(dim=1).numpy().astype("int")
            gt = mask_target_remove_no_json.detach().cpu().squeeze(1).numpy().astype("int")
            self.evaluator.add_batch(gt, pred)
            batch_miou = self.evaluator.Mean_Intersection_over_Union()
            self.evaluator.reset()

            # debug:
            # _=[]
            # for j in range(pred.shape[0]):
            #     _.append(self._cal_iou(pred[j],gt[j],args.classes,reduction="mean"))
            # _=sum(_)/len(_)
            # print(batch_miou)
            # print(_)
            # debug

            loss_mask_output_avg.update(loss_mask.item(), images_remove_no_json.size(0))
            # batch_time.update(time.time() - end)
            # end = time.time()

            if i % self.args.print_freq == 0:
                _lr = round(float(self.optimizer_.param_groups[0]['lr']), 6)
                _loss_mask = round(float(loss_mask_output_avg.avg), 6)

                logger.debug(f'\n[Training] | '
                             f'[{epoch}/{self.args.epochs}] | '
                             f'Loss mask:{_loss_mask} | lr:{_lr} | '
                             f'mIoU:{batch_miou}')

                self.writer.add_scalar("train_mask_loss", loss_mask_output_avg.avg, start_iter + i)

            if (i + 1) % self.args.save_freq == 0:
                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model_.state_dict(),
                        "best_acc1": 0.0,
                    }, False
                )
            i += 1

        self.evaluator.reset()

    def _load_label_map(self, path: str):
        with open(path, "r") as f:
            lines = f.readlines()
            idx = [int(l.strip().split("x")[0]) for l in lines]
            label = [l.strip().split("x")[1] for l in lines]
            label_map = dict(zip(label, idx))

        return label_map

    def _labelme2mask(self, args, json_path: str, label_map):
        if not os.path.exists(json_path):
            return np.zeros((args.input_h, args.input_w), dtype="uint8")

        with open(json_path, "r") as f:
            json_data = json.load(f)
        img_h = json_data["imageHeight"]
        img_w = json_data["imageWidth"]
        masks = [np.zeros((img_h, img_w), dtype="uint8") for i in range(args.classes)]
        for d in json_data["shapes"]:
            label = d["label"]
            points = np.array(d["points"], dtype=np.int32)
            idx = label_map[label]
            masks[idx] = cv2.drawContours(masks[idx], [points], -1, color=[255], thickness=-1)

        masks = [cv2.resize(x, (args.input_w, args.input_h)) for x in masks]

        resized_mask = np.zeros((args.input_h, args.input_w), dtype="uint8")
        for i, mask in enumerate(masks):
            if i == 0:
                continue
            resized_mask[mask == 255] = i

        return resized_mask

    def _cal_iou(self, pred, gt, num_class, reduction="mean"):
        # single image segmentation iou compute, only for checking config confliction
        assert reduction in ["none", "mean"]
        iou_scores = []
        for i in range(num_class):
            temp_mat_1 = np.zeros((pred.shape[0], pred.shape[1]), dtype="uint8")
            temp_mat_2 = np.zeros((pred.shape[0], pred.shape[1]), dtype="uint8")

            temp_mat_1[pred == i] = 1
            temp_mat_2[gt == i] = 1
            add_mat = temp_mat_1 + temp_mat_2

            intersection = (add_mat == 2).sum()

            union = temp_mat_1.sum() + temp_mat_2.sum() - intersection

            if (union == 0) and (intersection == 0):
                iou_scores.append(np.nan)
            else:
                iou_scores.append(intersection / union)

            # debug                
            if (union == 0) and (intersection > 0):
                print("weird!")
                raise

        if reduction == "none":
            return iou_scores
        else:
            return np.nanmean(iou_scores)
