import abc, shutil, os, time, socket
from datetime import datetime
from loguru import logger
from tqdm import tqdm
import torch
from torchvision import datasets
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from typing import Callable
from utils.evaluation import top1, top5, topk

ImageFile.LOAD_TRUNCATED_IMAGES = True

import network
from utils.meter import AverageMeter
from utils.meter import ProgressMeter
from utils.util import accuracy
from utils.util import get_key
from utils.util import make_dir
from utils.util import vis_maps
from lr_scheduler.lr_adjustment import LambdaLR
from loss.loss_builder import build_loss
from torch.optim.lr_scheduler import OneCycleLR


class BaseTrainer(abc.ABC):
    def __init__(self):
        self.args = None
        self.model_ = None
        self.optimizer_ = None
        self.train_loader_ = None
        self.val_loader_ = None
        self.lr_scheduler_ = None
        self.criterion = None
        self.writer = None

    def normalize_transpose(self, input):
        ...

    def init_trainer_v2(self,
                        args,
                        train_transforms: Callable,
                        val_transforms: Callable,
                        tensorboard_save_path: str = 'temp/tensorboard',
                        **kwargs):
        # self.args = kwargs["args"]
        self.args = args
        self.model_ = self.create_model(self.args.net,
                                        self.args.pretrained,
                                        num_classes=self.args.classes).cuda(self.args.gpus)
        self.optimizer_ = self.define_optimizer(self.args.lr)
        self.train_loader_ = self.define_loader(self.args.train_data_path,
                                                train_transforms,
                                                batch_size=self.args.batch_size,
                                                num_workers=self.args.workers,
                                                shuffle=True)
        self.val_loader_ = self.define_loader(self.args.val_data_path,
                                              val_transforms,
                                              batch_size=self.args.batch_size,
                                              num_workers=self.args.workers,
                                              shuffle=True)
        self.lr_scheduler_ = self.define_lr_scheduler(self.optimizer_)
        self.criterion = self.define_criterion(self.args.criterion_list, self.args.gpus)
        self.writer = self.define_scalar(tensorboard_save_path, comment=self.optimizer_.__module__)

        if isinstance(self.args.topk, (int, list, tuple)) is False:
            logger.error(f'args.topk type error,only support:(int,list, tuple) -> exit.')
            exit()

    def create_model(self, model_names: str, pretrained: bool, **kwargs):
        model = network.__dict__[model_names](pretrained=pretrained, **kwargs)
        return model

    def resume(self, model_path: str = "temp/weights/checkpoint.pth.tar", strict: bool = False, **kwargs):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            static_dict = checkpoint['state_dict']
            self.model_.load_state_dict(static_dict, strict=strict)
            print("{} 权重resume成功。".format(model_path))
        else:
            print("{} 权重文件不存在。".format(model_path))

    def save_checkpoint(self, state, is_best: bool, save_path: str = "temp/weights"):
        filename = os.path.join(save_path, "checkpoint.pth.tar")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            print("create path {} to save checkpoint.".format(save_path))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(save_path, "model_best.pth.tar"))

    def define_scalar(self, save_path: str = "temp/tensorboard", comment: str = "", **kwargs):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join(
            save_path, current_time + '_' + socket.gethostname() + comment)
        writer = SummaryWriter(logdir, write_to_disk=False)
        return writer

    def define_optimizer(self, lr: float, **kwargs):
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=lr)
        return optimizer

    def define_loader(self, path: str, trans: Callable, **kwargs):
        dataset_img = datasets.ImageFolder(path, transform=trans)
        data_loader = torch.utils.data.DataLoader(dataset_img, **kwargs)
        return data_loader

    def define_criterion(self, criterion_list: list, gpus: int, **kwargs):
        criterion = []
        for criterion_name in criterion_list:
            criterion.append(build_loss(criterion_name, **kwargs).cuda(gpus))
        return criterion

    # def define_lr_scheduler(self, optimizer):
    #     lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch / 4 + 1))
    #     return lr_scheduler

    def define_lr_scheduler(self, optimizer):
        lrs = OneCycleLR(optimizer,
                         max_lr=self.args.lr,
                         steps_per_epoch=len(self.train_loader_),
                         epochs=self.args.epochs,
                         pct_start=0.2)

        return lrs

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

    def train_epoch(self,
                    epoch: int,
                    normalize_transpose=None):
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

            # acc1, acc5 = accuracy(output, target, topk=self.args.topk)
            losses.update(loss.item(), images.size(0))

            for j in range(len(self.args.topk)):
                topn[j].update(acc[j], images.size(0))

            # top1.update(acc1[0], images.size(0))
            # topn.update(acc5[0], images.size(0))

            self.optimizer_.zero_grad()
            loss.backward()
            self.optimizer_.step()
            self.lr_scheduler_.step()

            if i % self.args.print_freq == 0:
                # 这里进行记录，是为了避免过多的数据跳动。
                origin_input = normalize_transpose(images)
                origin_input = np.array(origin_input.cpu().detach().numpy(), dtype=np.uint8)
                self.writer.add_images('input_img', origin_input, (start_iter + i))
                self.writer.add_scalar("train_loss", losses.avg, (start_iter + i))
                # self.writer.add_scalar("train_acc1", top1.avg, (start_iter + i))
                for k in range(len(self.args.topk)):
                    self.writer.add_scalar(f"train_acc{k}", topn[k].avg, (start_iter + i))

                _lr = round(float(self.optimizer_.param_groups[0]['lr']), 6)
                _loss = round(float(losses.avg), 6)
                # _acc1 = round(float(top1.avg), 4)
                topn_str = ''
                for k in topn:
                    topn_str += f' | {k.name}:{round(float(k.avg), 4)}'

                logger.debug(f'[Training] | '
                             f'[{epoch}/{self.args.epochs}] | '
                             f'Loss:{_loss} | '
                             f'lr:{_lr} | ' +
                             topn_str)

    def test_mask(self,
                  show_channel: int,
                  score_threshold: float,
                  area_threshold: float,
                  args,
                  test_error_src_img_save_path: str = "temp/origin_hard",
                  test_res_img_save_path: str = "temp/output",
                  img_mask_info_save_file: str = "temp//img_mask_info_save_file.txt",
                  mask_res_file: str = "temp//mask.txt",
                  **kwargs):
        """
            Test the network segmentation effect use val dataset.
            只适用于单通道的目标分割。
        Args:
            epoch：当前测试的epoch
            show_channel: 使用的网络输出的第几通道，用于检测效果。
            score_threshold: 输出的特征图，判断是否是目标点的分数阈值
            area_threshold: 对输出特征图
            args: 配置文件里的配置参数
            test_error_src_img_save_path:  误检原始数据存放路径
            test_res_img_save_path:  误检原始数据的误检效果图
            img_mask_info_save_file: 测试集所有图片推理结果信息存储文件，用于后期通过脚本确定最优分割阈值
            mask_res_file: 每一类别的误检类别统计数据。
            **kwargs:

        Returns:

        """
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        loss_mask_output_avg = AverageMeter('loss_test_mask', ':.4e')
        progress = ProgressMeter(
            len(self.val_loader_),
            [batch_time, data_time, loss_mask_output_avg])
        img_mask_info_fs = open(img_mask_info_save_file, 'w')
        mask_res_fs = open(mask_res_file, 'w')
        make_dir(test_error_src_img_save_path)
        make_dir(test_res_img_save_path)
        make_dir(test_error_src_img_save_path)

        test_img_save_path = test_res_img_save_path + "//{}.png"

        sum_nums_dict = {}
        error_part_dict = {}
        self.model_.eval()
        end = time.time()
        with torch.no_grad():
            for i, items in enumerate(self.val_loader_):
                data_time.update(time.time() - end)
                if isinstance(items, list):
                    if len(items) >= 4:
                        (images, target, _, path) = items[:4]
                    else:
                        raise RuntimeError("length need greater than or equal 4")
                else:
                    raise RuntimeError("iter is not list!")
                if torch.cuda.is_available():
                    images = images.cuda(args.gpus)
                    target = target.cuda(args.gpus, non_blocking=True)

                # 进行加mask分支训练
                if torch.cuda.is_available():
                    images = images.cuda(args.gpus)

                output = self.model_(images)
                # 增加适用性，测试的时候可以制定需要进行测试的是哪部分值。
                if isinstance(output, list):
                    output = output[1]
                    if isinstance(output, list):
                        output = output[show_channel]
                batch_time.update(time.time() - end)
                end = time.time()
                if True:
                    # 根据阈值判断mask是否为瑕疵图，然后统计误割数量比
                    mask = torch.where(output < score_threshold, torch.zeros(1).cuda(), output)
                    mask_val_sum = torch.sum(mask, dim=(3)).sum(2).sum(1)

                    for idx, mask_area in enumerate(mask_val_sum):
                        # 存储每张相关的信息
                        target_id = str(int(target[idx].cpu().detach().numpy()))
                        target_name = get_key(self.val_loader_.dataset.class_to_idx, int(int(target_id)))
                        img_mask_info_fs.writelines("{} {} {} {}\n".format(path[idx], target_id, target_name[0],
                                                                           mask_val_sum[
                                                                               idx].cpu().detach().numpy()))
                    mask_error_flag = mask_val_sum > area_threshold
                    for idx, mask_error_flag_item in enumerate(mask_error_flag):
                        # 这里进行存储每一个类别的路径。
                        target_id = str(int(target[idx].cpu().detach().numpy()))
                        per_cls_sum_img_list = sum_nums_dict.get(target_id)
                        if per_cls_sum_img_list:
                            per_cls_sum_img_list.append(path[idx])
                            sum_nums_dict.update({target_id: per_cls_sum_img_list})
                        else:
                            sum_nums_dict.update({target_id: [path[idx]]})

                        if mask_error_flag_item:
                            # 测试输出结果：
                            output_cpu = output.cpu().detach().numpy()
                            cur_img = images[idx]
                            img_name = os.path.basename(path[idx])
                            cur_img_save_path = test_img_save_path.format(
                                "{}".format(img_name[:-4]) + ("_{}".format(target[idx])))
                            mask_target = output_cpu[idx][0] * 255
                            mask_target_rgb = cv2.cvtColor(mask_target, cv2.COLOR_GRAY2BGR)
                            mask_target_rgb = np.array(mask_target_rgb, dtype=np.uint8)
                            cur_img_cpu = ((cur_img.cpu().detach().numpy() * 0.224) + 0.456) * 255
                            cur_img_cpu = cv2.cvtColor(np.array(cur_img_cpu.transpose(1, 2, 0), dtype=np.uint8),
                                                       cv2.COLOR_RGB2BGR)
                            dst = cv2.addWeighted(cur_img_cpu, 0.7,
                                                  mask_target_rgb, 0.3, 0)
                            cur_test_img_save_path = test_img_save_path.format(
                                "{}".format(img_name[:-4]) + (
                                    "_{}_a_area_{:.1f}".format(target[idx], mask_val_sum[idx])))
                            cur_test_src_img_save_path = test_img_save_path.format(
                                "{}".format(img_name[:-4]) + ("_{}_org".format(target[idx])))
                            cv2.imwrite(cur_test_img_save_path, dst)
                            cv2.imwrite(cur_img_save_path, output_cpu[idx][0] * 255)
                            cv2.imwrite(cur_test_src_img_save_path, cur_img_cpu)
                            # 进行剪切误割的图片到某个文件夹里。进行增加多一个瓶身工位文件夹
                            img_list = error_part_dict.get(target_id)
                            if img_list:
                                img_list.append(path[idx])
                                error_part_dict.update({target_id: img_list})
                            else:
                                error_part_dict.update({target_id: [path[idx]]})

                            cur_error_save_path = test_error_src_img_save_path + "\\" + path[idx].split("\\")[-2]
                            if not os.path.exists(cur_error_save_path):
                                os.mkdir(cur_error_save_path)
                            try:
                                shutil.copy(path[idx], cur_error_save_path)
                            except:
                                continue
                if i % args.print_freq == 0:
                    progress.display(i)
                i += 1

        sum_error_nums = 0
        sum_test_nums = 0
        for key, val in sum_nums_dict.items():
            cur_cls_sum_nums = len(set(sum_nums_dict.get(key, None)))
            error_path_list = error_part_dict.get(key, 0)
            part_name = get_key(self.val_loader_.dataset.class_to_idx, int(key))
            if error_path_list:
                error_path_list = set(error_path_list)
                error_nums = len(error_path_list)
                sum_error_nums += error_nums
                scale = 0.0
                if cur_cls_sum_nums != 0:
                    scale = error_nums / cur_cls_sum_nums
            else:
                error_nums = 0.0
                scale = 0.0
            sum_test_nums += cur_cls_sum_nums
            mask_res_fs.writelines(
                "部位：{} 误割数量：{} 总部位数：{} 误割比例：{}\n".format(part_name[0], error_nums, cur_cls_sum_nums,
                                                                       scale))
        mask_res_fs.close()
        img_mask_info_fs.close()

    def test_img(self,
                 show_channel: int,
                 score_threshold: float,
                 area_threshold: float,
                 img_path_list: list,
                 transforms,
                 root_path: str,
                 input_w: int,
                 input_h: int,
                 gpus: int,
                 output_save_path: str = "temp/output"):
        """
        主要用在单通道的前景和背景目标的分割测试。偏向于为了测试误检。
        Args:
            show_channel:
            score_threshold:
            area_threshold:
            img_path_list:
            transforms:
            root_path:
            input_w:
            input_h:
            gpus:
            output_save_path:

        Returns:

        """
        make_dir(output_save_path)
        cost_time = []
        error_mask_num = 0
        self.model_.eval()
        with torch.no_grad():
            for img_path in tqdm.tqdm(img_path_list):
                test_img_add_save_path = output_save_path + "\\{}_add.png".format(img_path[:-4])
                img_src_path = os.path.join(root_path, img_path)
                img = cv2.imread(img_src_path)
                img = cv2.resize(img, (input_h, input_w))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_color = Image.fromarray(img)
                inputs = transforms(img_color)
                inputs = inputs.unsqueeze(0)
                inputs = inputs.cuda(gpus)
                start_time = time.time()
                predict = self.model_(inputs)
                if isinstance(predict, list):
                    predict = predict[show_channel]
                cost_time.append(time.time() - start_time)
                if True:
                    mask_new = torch.where(predict < score_threshold, torch.zeros(1).cuda(gpus), predict)
                    mask_new_sum = torch.sum(mask_new, dim=(3)).sum(2).sum(1)
                    mask_new_max_thre = mask_new_sum > area_threshold
                    if mask_new_max_thre:
                        print("出现误割。", img_path)
                        error_mask_num += 1
                if mask_new_max_thre:
                    mask_output_cpu = predict.cpu().detach().numpy()
                    test_img_save_path = output_save_path + "\\{}_{}.png".format(img_path, int(
                        mask_new_sum.cpu().detach().numpy()))
                    cv2.imwrite(test_img_save_path, mask_output_cpu[0][0] * 255)
                    mask_target = mask_output_cpu[0][0] * 255
                    mask_target_rgb = cv2.cvtColor(mask_target, cv2.COLOR_GRAY2BGR)
                    mask_target_rgb = np.array(mask_target_rgb, dtype=np.uint8)
                    dst = cv2.addWeighted(img, 0.7, mask_target_rgb, 0.3, 0)
                    cv2.imwrite(test_img_add_save_path, dst)
                    shutil.copy(img_src_path, output_save_path)
            print("误割数据量为{}，比例为：{}".format(error_mask_num, error_mask_num / len(img_path_list)))

    def test_full_segmentation(self,
                               transforms,
                               root_path: str,
                               input_w: int,
                               input_h: int,
                               gpus: int,
                               classes: int,
                               output_save_path: str = "temp/output"):
        self.model_.eval()
        cost_time = []
        with torch.no_grad():
            for dir_root, dirs, files in os.walk(root_path):
                for file in files:
                    if os.path.splitext(file)[1] in [".jpg", ".png", ".bmp"]:
                        file = os.path.join(dir_root, file)
                        img = Image.open(file)
                        img_pil = img.resize((input_w, input_h), Image.BILINEAR)
                        img = transforms(img_pil)
                        input = img.unsqueeze(0).cuda(gpus)
                        start_time = time.time()
                        predict = self.model_(input)
                        if isinstance(predict, list):
                            predict = predict[1]
                            if isinstance(predict, list):
                                predict = predict[0]
                        cost_time.append(time.time() - start_time)
                        mask = predict.squeeze(0).cpu().numpy().argmax(0)
                        basename = os.path.basename(file)
                        save_name = os.path.join(output_save_path, basename)
                        vis_maps(img_pil, mask, classes, save_name)

            print("平均耗时：{}".format(np.array(cost_time).sum() / len(cost_time)))

    def validate(self, epoch: int, **kwargs):
        """
        此函数只验证分类准确度、分类的损失。其也可以用于多任务的类别分支的准确度预测。
        Args:
            top_k: 进行验证的类别top几分数
            args: 进行预测的参数配置
        Returns:
        """
        losses = AverageMeter('Loss', ':.4e')

        if isinstance(self.args.topk, (list, tuple)):
            topn = [AverageMeter(f'Acc@{k}') for k in self.args.topk]
        else:
            topn = [AverageMeter(f'Acc@{self.args.topk}')]

        self.model_.eval()

        with torch.no_grad():
            # end = time.time()
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
                losses.update(loss.item(), images.size(0))

                acc = topk(output, target, k=self.args.topk)
                for j in range(len(self.args.topk)):
                    topn[j].update(acc[j], images.size(0))

                topn_str = ''
                for n in topn:
                    topn_str += f' | {n.name}:{round(float(n.avg), 4)}'

                if i % self.args.print_freq == 0:
                    logger.debug(f'[Val] | '
                                 f'{losses.name}: {round(float(losses.avg), 4)}' +
                                 topn_str)

        self.writer.add_scalar("val_acc1", topn[0].avg, epoch)

        return topn[0].avg

    def judge_cycle_mask_train(self, output, mask, area_threshold=100, judge_channel: int = 0, target_id: int = 0):
        """
        使用focal loss的思想，降低误踢
        Args:
            output: 网络的输出值
            mask: mask掩膜
            judge_channel: 对于list的输出，进行选择对于的channel。
            target_id: 背景的id，如果背景的id值输出灰度大于阈值则进行不符合判断。

        Returns:

        """
        if isinstance(output, list):
            cur_output = output[judge_channel]
        else:
            cur_output = output
        cur_output = cur_output[mask == target_id]
        if cur_output.sum() < area_threshold:
            return False
        else:
            # print(cur_output.sum())
            return True

    # def export_torchscript(self, args, save_path):
    #     print("export torchscript...")
    #     rand_input = torch.rand(1, 3, args.input_h, args.input_w).cuda()
    #
    #     from export.shufflenetv2 import shufflenet_v2_x1_0
    #
    #     model = shufflenet_v2_x1_0(num_classes=args.classes)
    #     model_path = args.data.replace("config", "models/checkpoint.pth.tar")
    #     checkpoint = torch.load(model_path)
    #     static_dict = checkpoint['state_dict']
    #     model.load_state_dict(static_dict, strict=True)
    #
    #     model.cuda()
    #     model.eval()
    #
    #     torchscript = torch.jit.trace(model, rand_input, strict=False)
    #     localtime = time.localtime(time.time())
    #     date = "-".join([str(i) for i in localtime[0:3]])
    #
    #     file_name = "{}_{}_{}x{}_{}.torchscript.pt".format(args.model_names, \
    #                                                        str(args.classes), \
    #                                                        str(args.input_w), \
    #                                                        str(args.input_h), \
    #                                                        date)
    #     torchscript.save(os.path.join(save_path, file_name))
    #     print("ok")

    @torch.no_grad()
    def check_data_conflict(self, args, transforms, tag: str = "train"):
        # only check ok and ng conflicts
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
            pred_cls_id = predict.softmax(dim=1).argmax(dim=1).item()
            if (gt_id > 0 and pred_cls_id == 0) or (gt_id == 0 and pred_cls_id > 0):
                # ng predicted as ok || ok predicted as ng
                dst_path = os.path.join(args.data, "conflict-" + tag, gt)
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                shutil.move(p, dst_path)

            # if (not args.control.projectOperater.checking_train) and \
            #         (not args.control.projectOperater.checking_test):
            #     print("abort")
            #     return
            #     # ui display progress
            # args.control.projectOperater.dataCount += 1 / len(image_paths)

        print("done")

    def _get_images(self, path: str):
        ret = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_extend = file.split(".")[-1]
                if file_extend in ["jpg", "png", "jpeg"]:
                    ret.append(os.path.join(root, file))
        return ret
