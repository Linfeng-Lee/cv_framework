import abc, shutil, os, time, socket
from datetime import datetime
from typing import Union, Callable

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from tensorboardX import SummaryWriter
from PIL import Image, ImageFile
from loguru import logger
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR

from lr_scheduler.lr_adjustment import LambdaLR
from utils.util import get_time

ImageFile.LOAD_TRUNCATED_IMAGES = True

import network
from utils.meter import AverageMeter
from utils.meter import ProgressMeter
from utils.util import accuracy, load_weight
from utils.util import get_key
from utils.util import make_dir
from utils.util import vis_maps
# from utils.util import get_image
from utils.evaluation import top1, top5, topk
from loss.loss_builder import build_loss
from loss.center_loss import CenterLoss
from base_impl import ImplBaseTrainer


class EmbeddingTrainer(ImplBaseTrainer, abc.ABC):
    def __init__(self):
        super(EmbeddingTrainer, self).__init__()
        self.center_loss_weight = None
        self.center_loss = None
        self.losses = None
        self.test_loader_ = None
        self.val_data_path = None
        self.train_data_path = None

    def init_trainer(self,
                     args,
                     train_transforms: Callable,
                     val_transforms: Callable,
                     tensorboard_save_path: str):
        self.args = args

        self.model_ = self.create_model(self.args.net,
                                        self.args.pretrained,
                                        embedding_classes=self.args.classes).cuda(self.args.gpus)
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
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.args.input_h, self.args.input_w)),
            transforms.Normalize(mean=self.args.augment.mean, std=self.args.augment.std)
        ])
        self.test_loader_ = self.define_loader(self.args.val_data_path,
                                               transform=test_transforms,
                                               batch_size=self.args.test_batch_size,
                                               num_workers=self.args.workers)
        self.criterion = self.define_criterion(self.args.criterion_list, self.args.gpus)
        self.writer = self.define_scalar(tensorboard_save_path, comment=self.optimizer_.__module__)

        self.center_loss = CenterLoss(num_classes=self.args.classes, feat_dim=1024)
        self.center_loss_weight = 0.5
        self.optimizer_.add_param_group({'params': self.center_loss.parameters(),
                                         'lr': self.args.lr})
        self.lr_scheduler_ = self.define_lr_scheduler(self.optimizer_,
                                                      max_lr=self.args.lr,
                                                      steps_per_epoch=len(self.train_loader_),
                                                      epochs=self.args.epochs,
                                                      pct_start=0.2)
        if isinstance(self.args.topk, (int, list, tuple)) is False:
            logger.error(f'args.topk type error,only support:(int,list, tuple) -> exit.')
            exit()

    def save_checkpoint(self, state: dict, save_path: str, **kwargs):
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

        state["center_loss"] = self.center_loss.state_dict()

        torch.save(state, filename)

        is_best = kwargs.get('is_best')
        if is_best is not None:
            # e.g.2022-11-14-(17:35:42)_model_best.pth.tar
            shutil.copyfile(filename, os.path.join(save_path, f"{curr_time}_model_best.pth.tar"))

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
            logger.debug(f'[Training] | '
                         f'[{epoch}/{self.args.epochs}] | '
                         f'Acc@1:{top1_acc} | '
                         f'Best Acc@1:{best_acc}')

    def train_epoch(self, epoch: int, **kwargs):
        if isinstance(self.args.topk, (list, tuple)):
            topn = [AverageMeter(f'Acc@{k}') for k in self.args.topk]
        else:
            topn = [AverageMeter(f'Acc@{self.args.topk}')]

        losses = AverageMeter(f'loss')

        self.model_.train()

        train_len = len(self.train_loader_)
        start_iter = epoch * train_len
        self.writer.add_scalar("lr", self.optimizer_.param_groups[0]['lr'], epoch)

        pbar = tqdm(enumerate(self.train_loader_), total=train_len, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (images, target) in pbar:
            if torch.cuda.is_available():
                images = images.cuda(self.args.gpus)
                target = target.cuda(self.args.gpus, non_blocking=True)

            output, output_emb = self.model_(images)

            binary_target = torch.zeros_like(target)
            binary_target[target > 0] = 1
            loss_cls = self.criterion[0](output, binary_target)
            loss_emb = self.criterion[0](output_emb, target)
            loss_cl = self.center_loss(self.model_.embedding, target)

            loss = loss_cls + (loss_emb + self.center_loss_weight * loss_cl)
            self.optimizer_.zero_grad()
            loss.backward()

            # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
            lr = self.optimizer_.param_groups[0]['lr']
            lr_cent = self.optimizer_.param_groups[1]['lr']
            for param in self.center_loss.parameters():
                param.grad.data *= (lr_cent / (self.center_loss_weight * lr))

            self.optimizer_.step()
            self.lr_scheduler_.step()

            # compute topk
            acc = topk(output, binary_target, k=self.args.topk)

            losses.update(loss.item(), images.size(0))
            self.losses = round(float(losses.avg), 6)
            for j in range(len(self.args.topk)):
                topn[j].update(acc[j], images.size(0))

            if i % self.args.print_freq == 0:
                normalize_transpose = kwargs.get('normalize_transpose')
                origin_input = normalize_transpose(images)
                origin_input = np.array(origin_input.cpu().detach().numpy(), dtype=np.uint8)
                self.writer.add_images('input_img', origin_input, (start_iter + i))
                self.writer.add_scalar("train_loss", losses.avg, (start_iter + i))
                # self.writer.add_scalar("train_acc1", top1.avg, (start_iter + i))
                for k in range(len(self.args.topk)):
                    self.writer.add_scalar(f"train_acc{k}", topn[k].avg, (start_iter + i))

                _lr = round(float(self.optimizer_.param_groups[0]['lr']), 6)
                _loss = round(float(losses.avg), 6)

                # e.g. top_k=(1,5) -> Acc@1:99.98 | Acc@5: 100:00
                topn_str = ''
                for k in topn:
                    topn_str += f' | {k.name}:{round(float(k.avg), 4)}'

                logger.debug(f'[Training] | '
                             f'[{epoch}/{self.args.epochs}] | '
                             f'{losses.name}:{_loss} | '
                             f'lr:{_lr}' +
                             topn_str)

    def validate(self, epoch: int, **kwargs):
        """
        此函数只验证分类准确度、分类的损失。其也可以用于多任务的类别分支的准确度预测。
        Args:
            top_k: 进行验证的类别top几分数
            args: 进行预测的参数配置
        Returns:
        """

        if isinstance(self.args.topk, (list, tuple)):
            topn = [AverageMeter(f'Acc@{k}') for k in self.args.topk]
        else:
            topn = [AverageMeter(f'Acc@{self.args.topk}')]

        losses = AverageMeter('Loss', )

        self.model_.eval()

        with torch.no_grad():
            pbar = tqdm(enumerate(self.train_loader_),
                        total=len(self.val_loader_),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for i, sample in pbar:
                if len(sample) == 2:
                    images, target = sample
                else:
                    images, target = sample[:2]

                if torch.cuda.is_available():
                    images = images.cuda(self.args.gpus)
                    target = target.cuda(self.args.gpus, non_blocking=True)

                output, output_emb = self.model_(images)

                binary_target = torch.zeros_like(target)
                binary_target[target > 0] = 1
                loss_cls = self.criterion[0](output, binary_target)
                loss_emb = self.criterion[0](output_emb, target)
                loss_cl = self.center_loss(self.model_.embedding, target)

                loss = loss_cls + (loss_emb + self.center_loss_weight * loss_cl)
                losses.update(loss.item(), images.size(0))

                acc = topk(output, binary_target, k=self.args.topk)
                for j in range(len(self.args.top_k)):
                    topn[j].update(acc[j], images.size(0))

                topn_str = ''
                for n in topn:
                    topn_str += f' | {n.name}:{round(float(n.avg), 4)}'

                # display
                if i % self.args.print_freq == 0:
                    logger.debug(f'[Val] | '
                                 f'{losses.name}: {round(float(losses.avg), 4)}' +
                                 topn_str)
            # display
            topn_str = ''
            for n in topn:
                topn_str += f' | {n.name}:{round(float(n.avg), 4)}'
            logger.info(f'[Val] | ' + topn_str)

        self.writer.add_scalar("val_acc1", topn[0].avg, epoch)

        return topn[0].avg

    def test(self, **kwargs):
        top_k = self.args.topk
        if isinstance(top_k, (list, tuple)):
            topn = [AverageMeter(f'Acc@{k}') for k in top_k]
        else:
            topn = [AverageMeter(f'Acc@{top_k}')]

        # images = get_image(self.val_data_path)

        self.model_.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(self.test_loader_), total=len(self.test_loader_))
            for i, sample in pbar:
                if len(sample) == 2:
                    images, target = sample
                else:
                    images, target = sample[:2]

                if torch.cuda.is_available():
                    images = images.cuda(self.args.gpus)
                    target = target.cuda(self.args.gpus, non_blocking=True)

                # gs_score = output.softmax(dim=1)[0][0].detach().cpu().item()
                # print(gs_score)
                # emb = output_emb[0].detach().cpu().numpy()
                # exit()

                binary_target = torch.zeros_like(target)
                binary_target[target > 0] = 1
                outputs, output_emb = self.model_(images)

                output = outputs[0].cpu()
                idx = torch.argmax(output, dim=0)
                pred = output[idx].softmax(0).cpu().item()
                binary_target_ = binary_target.cpu().item()

                acc = topk(outputs, binary_target, k=top_k)

                for j in range(len(top_k)):
                    topn[j].update(acc[j], images.size(0))
                topn_str = ''
                for n in topn:
                    topn_str += f' | {n.name} : {round(float(n.avg), 4)}'
                # torch.cuda.synchronize()
            logger.debug(f'[Test]' + topn_str)

    def _preprocess(self, image):
        # preprocess
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        transforms_ = transforms.Compose([
            transforms.Resize((self.args.input_h, self.args.input_w)),
            transforms.ToTensor(),
            normalize,
        ])
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_color = Image.fromarray(img)
        inputs = transforms_(img_color)
        inputs = inputs.unsqueeze(0)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        return inputs
