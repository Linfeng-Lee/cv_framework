import time

import torch
from torch import nn

from tool.trainer.base_trainer import BaseTrainer
from dataset.data_pipe import get_loader_mask, get_loader_class

from utils.meter import AverageMeter
from utils.meter import ProgressMeter
from utils.util import accuracy


class MultiTaskTrainer(BaseTrainer):
    def __init__(self):
        super(MultiTaskTrainer, self).__init__()
        self.train_loader_remove_no_json_ = None

    def init_trainer(self, model_names, lr, train_data_path, val_data_path,
                     gpus, classes, classify_transform,criterion_list=None,
                     tensorboard_save_path='temp/tensorboard', pretrained=False,
                     judge_cycle_train=False,
                     **kwargs):
        self.model_ = self.create_model(model_names, pretrained, num_classes=classes,**kwargs).cuda(gpus)
        self.optimizer_ = self.define_optimizer(lr)
        self.train_loader_ = get_loader_class(train_data_path, classify_transform, kwargs["balance_n_classes"],
                                              kwargs["balance_n_samples"], kwargs["class_id_map_save_path"])
        self.train_loader_remove_no_json_= self.define_loader(train_data_path, **kwargs)
        self.val_loader_ = self.define_loader(val_data_path,**kwargs)
        self.lr_scheduler_ = self.define_lr_scheduler(self.optimizer_)
        self.criterion= self.define_criterion(criterion_list,gpus,**kwargs)
        self.writer = self.define_scalar(tensorboard_save_path, comment=self.optimizer_.__module__)
        self.judge_cycle_train=judge_cycle_train

    def define_loader(self, path,  **kwargs):
        data_loader = get_loader_mask(path, **kwargs)
        return data_loader


    def train_epoch(self, epoch, top_k, args,**kwargs):
        topn_str = "Acc@{}".format(top_k)
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss_class', ':.4e')
        loss_mask_output_avg=AverageMeter('loss_mask',':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        topn = AverageMeter(topn_str, ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader_),
            [batch_time, data_time, losses,loss_mask_output_avg, top1, topn],
            prefix="Epoch: [{}]".format(epoch))

        self.model_.train()
        start_iter=epoch*len(self.train_loader_)
        end = time.time()
        self.writer.add_scalar("lr",self.optimizer_.param_groups[0]['lr'],epoch)
        # for i, (images, target) in enumerate(self.train_loader_):
        i=0
        for (images, target), no_json_list in\
                zip(self.train_loader_,self.train_loader_remove_no_json_):
            #为了兼容后期返回的各种数量长度
            if isinstance(no_json_list,list):
                images_remove_no_json, target_remove_no_json, mask_target_remove_no_json=no_json_list[:3]
            else:
                raise RuntimeError("iter is not list!")

            data_time.update(time.time() - end)
            if torch.cuda.is_available():
                images = images.cuda(args.gpus)
                target = target.cuda(args.gpus, non_blocking=True)

            output, _ = self.model_(images)
            loss = self.criterion[0](output, target)

            try:
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()
            except Exception as e:
                print(e,"请检查config配置的classes是不是实际训练集的类别数。")

            acc1, acc5 = accuracy(output, target, topk=(1, top_k))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            topn.update(acc5[0], images.size(0))
            self.writer.add_scalar("train_loss",losses.avg,start_iter+i)
            self.writer.add_scalar("train_acc1",top1.avg,start_iter+i)

            #进行加mask分支训练
            if torch.cuda.is_available():
                images_remove_no_json = images_remove_no_json.cuda(args.gpus)
                target_remove_no_json = target_remove_no_json.cuda(args.gpus, non_blocking=True)
                mask_target_remove_no_json=mask_target_remove_no_json.cuda(args.gpus)

            for cur_time in range(args.cycle_train_times):
                output, mask_output = self.model_(images_remove_no_json)
                loss_cls = self.criterion[0](output, target_remove_no_json)
                loss_mask=self.criterion[1](mask_output,mask_target_remove_no_json)
                loss=loss_cls+loss_mask
                if cur_time>0:
                    loss=loss*0.1
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()
                #是否关闭这个判断，有些网络构建是不适合这个判断的，如果输出的通道数是大于1的需要关闭这个功能。
                if self.judge_cycle_train:
                    # 进行困难样本的重新训练
                    if not self.judge_cycle_mask_train(mask_output,mask_target_remove_no_json,args.area_threshold):
                        break

            loss_mask_output_avg.update(loss_mask.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()


            if i % args.print_freq == 0:
                progress.display(i)
                self.writer.add_scalar("train_mask_loss",loss_mask_output_avg.avg,start_iter+i)
            i+=1

