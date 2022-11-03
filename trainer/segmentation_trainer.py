import os
import json
import time
from matplotlib import lines

import torch
from torch import nn
from tqdm import tqdm
from loguru import logger
from trainer.base_trainer import BaseTrainer
from dataset.data_pipe import get_loader_mask

from utils.meter import AverageMeter
from utils.meter import ProgressMeter
from utils.meter import SegmentationEvaluator
import numpy as np
import shutil
from PIL import Image
import cv2
from torch.optim.lr_scheduler import OneCycleLR


class SegmantationTrainer(BaseTrainer):
    def __init__(self):
        super(SegmantationTrainer, self).__init__()
        self.train_loader_remove_no_json_ = None

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
                     **kwargs):
        self.args = kwargs["args"]
        self.model_ = self.create_model(model_names, pretrained, num_classes=classes).cuda(gpus)
        self.optimizer_ = self.define_optimizer(lr)
        self.train_loader_remove_no_json_ = self.define_loader(train_data_path, **kwargs)
        self.val_loader_ = self.define_loader(val_data_path, **kwargs)
        self.lr_scheduler_ = self.define_lr_scheduler(self.optimizer_)
        self.criterion = self.define_criterion(criterion_list, gpus, **kwargs)
        self.writer = self.define_scalar(tensorboard_save_path, comment=self.optimizer_.__module__)
        self.evaluator = SegmentationEvaluator(classes)

    def define_lr_scheduler(self, optimizer):
        lrs = OneCycleLR(optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader_remove_no_json_),
                         epochs=self.args.epochs,
                         pct_start=0.2)

        return lrs

    def define_loader(self, path, **kwargs):
        data_loader = get_loader_mask(path, **kwargs)
        return data_loader

    @torch.no_grad()
    def validate(self, epoch, top_k, args, **kwargs):
        # return 1.0
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        progress = ProgressMeter(
            len(self.val_loader_),
            [batch_time, data_time],
            prefix="Epoch: [{}]".format(epoch))

        self.model_.eval()
        start_iter = epoch * len(self.val_loader_)
        end = time.time()
        i = 0
        self.evaluator.reset()
        for no_json_list in self.val_loader_:  # img_tensor,class_id,mask_trg,img_path
            # 为了兼容后期返回的各种数量长度
            data_time.update(time.time() - end)

            if isinstance(no_json_list, list):
                images_remove_no_json, target_remove_no_json, mask_target_remove_no_json = no_json_list[:3]
            else:
                raise RuntimeError("iter is not list!")

            # 进行加mask分支训练
            if torch.cuda.is_available():
                images_remove_no_json = images_remove_no_json.cuda(args.gpus)
                mask_target_remove_no_json = mask_target_remove_no_json.cuda(args.gpus)

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

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            i += 1

        self.evaluator.reset()

        return 1.0

    def fit(self, start_epoch: int, epochs: int, top_k: int, args, save_path: str, **kwargs):
        best_acc = 0.5
        for epoch in range(start_epoch, epochs):
            # self.lr_scheduler_.step()
            self.train_epoch(epoch, top_k, args, **kwargs)
            top1_acc = self.validate(epoch, top_k, args)
            is_best = top1_acc >= best_acc
            best_acc = max(top1_acc, best_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model_.state_dict(),
                    "best_acc1": top1_acc,
                }, is_best, save_path
            )

    def train_epoch(self, epoch: int, top_k: int, args, **kwargs):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        loss_mask_output_avg = AverageMeter('loss_mask', ':.4e')
        progress = ProgressMeter(
            len(self.train_loader_remove_no_json_),
            [batch_time, data_time, loss_mask_output_avg],
            prefix="Epoch: [{}]".format(epoch))

        self.model_.train()
        train_len = len(self.train_loader_remove_no_json_)
        start_iter = epoch * train_len
        end = time.time()
        self.writer.add_scalar("lr", self.optimizer_.param_groups[0]['lr'], epoch)
        i = 0
        self.evaluator.reset()
        pbar = tqdm(self.train_loader_remove_no_json_,
                    total=train_len,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for no_json_list in pbar:  # img_tensor,class_id,mask_trg,img_path
            data_time.update(time.time() - end)
            # 为了兼容后期返回的各种数量长度
            if isinstance(no_json_list, list):
                images_remove_no_json, target_remove_no_json, mask_target_remove_no_json = no_json_list[:3]
            else:
                raise RuntimeError("iter is not list!")

            # 进行加mask分支训练
            if torch.cuda.is_available():
                images_remove_no_json = images_remove_no_json.cuda(args.gpus)
                mask_target_remove_no_json = mask_target_remove_no_json.cuda(args.gpus)
            for cur_time in range(args.cycle_train_times):
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
                if not self.judge_cycle_mask_train(mask_output, target_remove_no_json, args.area_threshold):
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
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                _lr = round(float(self.optimizer_.param_groups[0]['lr']), 6)
                _loss_mask = round(float(loss_mask_output_avg.avg), 6)

                logger.debug(f'[{epoch}/{self.args.epochs}] | Loss mask:{_loss_mask} | lr:{_lr} | mIoU:{batch_miou}')

                self.writer.add_scalar("train_mask_loss", loss_mask_output_avg.avg, start_iter + i)

            if (i + 1) % args.save_freq == 0:
                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model_.state_dict(),
                        "best_acc1": 0.0,
                    }, False
                )
            i += 1

        self.evaluator.reset()

    def _load_label_map(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
            idx = [int(l.strip().split("x")[0]) for l in lines]
            label = [l.strip().split("x")[1] for l in lines]
            label_map = dict(zip(label, idx))

        return label_map

    def _labelme2mask(self, args, json_path, label_map):
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

    @torch.no_grad()
    def check_data_conflict(self, args, transforms, tag="train"):
        print("check {} config conflict...".format(tag))
        if tag not in ["train", "test"]:
            raise ValueError("tag either be 'train' or 'test'!")
        self.model_.eval()

        assert os.path.exists(args.data.replace("config", "models/class_id_map.txt")), "class_id_map.txt not found."

        label_map = self._load_label_map(args.data.replace("config", "models/class_id_map.txt"))
        image_paths = self._get_images(os.path.join(args.data, tag))
        json_paths = [p.replace(".png", ".json") for p in image_paths]

        for p_i, p_j in zip(image_paths, json_paths):
            _ = p_i.split(os.sep)
            folder_name = _[_.index(tag) + 1]

            img = Image.open(p_i).convert('RGB')

            img_pil = img.resize((args.input_w, args.input_h), Image.BILINEAR)
            img = transforms(img_pil)
            input = img.unsqueeze(0).cuda(args.gpus)
            predict = self.model_(input)[0].softmax(dim=1).argmax(dim=1, keepdim=True).cpu().numpy()[0][0]
            gt = self._labelme2mask(args, p_j, label_map)

            # define conflict config by IOU score
            miou_score = self._cal_iou(predict, gt, args.classes, reduction="mean")

            # case1: mean of iou scores is lowwer than given threshold
            miou_thresh = 0.5
            if miou_score < miou_thresh:
                dst_path = os.path.join(args.data, "conflict-" + tag, folder_name)
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                shutil.move(p_i, dst_path)
                if os.path.exists(p_j):
                    shutil.move(p_j, dst_path)

        print("done")

    def export_torchscript(self, args, save_path):
        print("export torchscript...")
        rand_input = torch.rand(1, 3, args.input_h, args.input_w).cuda()

        from export.shufflenetv2_segmantation import shufflenet_v2_x1_0

        model = shufflenet_v2_x1_0(num_classes=args.classes)
        model_path = args.data.replace("config", "models/checkpoint.pth.tar")
        checkpoint = torch.load(model_path)
        static_dict = checkpoint['state_dict']
        model.load_state_dict(static_dict, strict=True)
        model.cuda()
        model.eval()

        torchscript = torch.jit.trace(model, rand_input, strict=False)
        localtime = time.localtime(time.time())
        date = "-".join([str(i) for i in localtime[0:3]])

        file_name = "{}_{}_{}x{}_{}.torchscript.pt".format(args.model_names, \
                                                           str(args.classes), \
                                                           str(args.input_w), \
                                                           str(args.input_h), \
                                                           date)
        torchscript.save(os.path.join(save_path, file_name))
        print("ok")

    def export_onnx(self, args, save_path):
        print("export onnx...")
        rand_input = torch.rand(1, 3, args.input_h, args.input_w).cpu()

        from export.shufflenetv2_segmantation import shufflenet_v2_x1_0

        model = shufflenet_v2_x1_0(num_classes=args.classes)
        model_path = args.data.replace("config", "models/checkpoint.pth.tar")
        checkpoint = torch.load(model_path)
        static_dict = checkpoint['state_dict']
        model.load_state_dict(static_dict, strict=True)
        model.cpu()
        model.eval()

        localtime = time.localtime(time.time())
        date = "-".join([str(i) for i in localtime[0:3]])

        file_name = "{}_{}_{}x{}_{}.onnx".format(args.model_names, \
                                                 str(args.classes), \
                                                 str(args.input_w), \
                                                 str(args.input_h), \
                                                 date)
        torch.onnx.export(model, rand_input, os.path.join(save_path, file_name), verbose=False, opset_version=12,
                          input_names=['images'],
                          output_names=['output'])

        print("ok")
