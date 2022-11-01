import time

import torch
from torch import nn
import numpy as np

from tool.trainer.base_trainer import BaseTrainer
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

    def init_trainer(self, model_names, lr, train_data_path, val_data_path,
                     gpus, classes, classify_transform, criterion_list=None,
                     tensorboard_save_path='temp/tensorboard', pretrained=False,
                     judge_cycle_train=False,
                     **kwargs):
        self.epoches=kwargs["epoches"]
        self.args=kwargs["args"]
        self.lr=lr
        self.model_ = self.create_model(model_names, pretrained, num_classes=classes).cuda(gpus)
        self.optimizer_ = self.define_optimizer(lr)
        self.classify_transform = classify_transform
        self.train_loader_ = get_loader_multi_label_class(train_data_path, classify_transform, kwargs["balance_n_classes"],
                                              kwargs["balance_n_samples"], kwargs["class_id_map_save_path"], kwargs["tensor_transform"], kwargs["weights"])
        self.val_loader_ = get_loader_multi_label_class(val_data_path, kwargs["tensor_transform"], kwargs["balance_n_classes"],
                                              kwargs["balance_n_samples"], kwargs["class_id_map_save_path"], kwargs["tensor_transform"], kwargs["weights"], is_training=False)
        self.lr_scheduler_ = self.define_lr_scheduler(self.optimizer_)
        self.criterion= self.define_criterion(criterion_list,gpus,**kwargs)
        self.writer = self.define_scalar(tensorboard_save_path, comment=self.optimizer_.__module__)
        self.judge_cycle_train=judge_cycle_train


    def fit(self, start_epoch, epochs, top_k, args,save_path, **kwargs):
        best_acc = 0.5
        for epoch in range(start_epoch, epochs):
            # self.lr_scheduler_.step()
            self.train_epoch(epoch, top_k, args,**kwargs)
            top1_acc = self.validate(epoch, top_k, args)
            is_best = top1_acc >= best_acc
            best_acc=max(top1_acc,best_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model_.state_dict(),
                    "best_acc1": top1_acc,
                }, is_best,save_path
            )
            
            if (not args.control.projectOperater.automate) and \
                (not args.control.projectOperater.training):
                break     


    def define_lr_scheduler(self, optimizer):
        lrs = OneCycleLR(optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader_), epochs=self.args.epochs,
                                        pct_start=0.2)

        return lrs

    def train_epoch(self, epoch, top_k, args,**kwargs):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss_class', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        topmulti = AverageMeter("accmulti", ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader_),
            [batch_time, data_time, losses, top1, topmulti],
            prefix="Epoch: [{}]".format(epoch))

        self.model_.train()
        start_iter=epoch*len(self.train_loader_)
        end = time.time()
        self.writer.add_scalar("lr",self.optimizer_.param_groups[0]['lr'],epoch)
        for i, (images, target) in enumerate(self.train_loader_):

            data_time.update(time.time() - end)

            if torch.cuda.is_available():
                images = images.cuda(args.gpus)
                target = target.cuda(args.gpus, non_blocking=True)

            output = self.model_(images)
            
            loss = self.criterion[0](output, target)

            try:
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()
                self.lr_scheduler_.step()

            except Exception as e:
                print(e,"请检查config配置的classes是不是实际训练集的类别数。")

            acc1, accmulti = accuracymultilabel(output, target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            topmulti.update(accmulti[0], images.size(0))

            #ui display training progress
            args.control.projectOperater.dataCount += 1/(args.epochs* len(self.train_loader_))
            
            
            if i % args.print_freq == 0:
                # 这里进行记录，是为了避免过多的数据跳动。
                self.writer.add_scalar("train_loss",losses.avg,start_iter+i)
                self.writer.add_scalar("train_acc1",top1.avg,start_iter+i)
                progress.display(i)
                print("lr:",self.optimizer_.param_groups[0]['lr'],"\n")

                args.control.projectOperater.loss=losses.avg
                args.control.projectOperater.trainAcu=top1.avg

            if (not args.control.projectOperater.automate) and \
                                (not args.control.projectOperater.training):
                break                                        


    def validate(self, epoch, top_k, args, **kwargs):
        """
        此函数只验证分类准确度、分类的损失。其也可以用于多任务的类别分支的准确度预测。
        Args:
            top_k: 进行验证的类别top几分数
            args: 进行预测的参数配置
        Returns:
        """

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        topmulti = AverageMeter("accmulti", ':6.2f')
        progress = ProgressMeter(
            len(self.val_loader_),
            [batch_time, losses, top1, topmulti],
            prefix='Test: ')

        confusion_matrix_multilabel = np.zeros((2*args.classes, 2*args.classes), dtype=np.long)

        self.model_.eval()

        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate(self.val_loader_):
                if len(sample) == 2:
                    images, target = sample
                else:
                    images, target = sample[:2]

                if torch.cuda.is_available():
                    images = images.cuda(args.gpus)
                    target = target.cuda(args.gpus, non_blocking=True)

                output = self.model_(images)
                if isinstance(output, list):
                    output = output[0]

                loss = self.criterion[0](output, target)

                acc1, accmulti = accuracymultilabel(output, target)
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                topmulti.update(accmulti[0], images.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                output[output > 0.5] = 1
                output[output <= 0.5] = 0

                for t, p in zip(target.long(), output.long()):
                    for j, (t0, p0) in enumerate(zip(t, p)):
                        confusion_matrix_multilabel[2 * j + 1 - t0, 2 * j + 1 - p0] += 1

                if i % args.print_freq == 0:
                    progress.display(i)
                    
                if (not args.control.projectOperater.automate) and \
                                    (not args.control.projectOperater.testing) and (not args.control.projectOperater.training):
                    break          
            # TODO: 返回top几的准确度
            print(' * Acc@1 {top1.avg:.3f} accmulti {topmulti.avg:.3f}'
                  .format(top1=top1, topmulti=topmulti))

        print(confusion_matrix_multilabel)
        self.writer.add_scalar("val_acc1", top1.avg, epoch)
        args.control.projectOperater.testAcu=top1.avg

        return top1.avg

    @torch.no_grad()
    def check_data_conflict(self,args,transforms,tag="train"):
        print("check {} params conflict...".format(tag))
        if tag not in ["train","test"]:
            raise ValueError("tag either be 'train' or 'test'!")
        self.model_.eval()
        label_map = sorted(os.listdir(os.path.join(args.data,tag)))
        image_paths = self._get_images(os.path.join(args.data,tag))
        for p in image_paths:
            _=p.split(os.sep)
            gt=_[_.index(tag)+1]
            gt_id = label_map.index(gt)
            img=Image.open(p).convert('RGB')

            img_pil=img.resize((args.input_w,args.input_h),Image.BILINEAR)
            img=transforms(img_pil)
            input=img.unsqueeze(0).cuda(args.gpus)
            predict=self.model_(input)

            pred_gs=predict.sigmoid().squeeze()[0].item()
            pred_cls_id=0 if pred_gs > 0.5 else 1

            pred_score = predict.sigmoid().squeeze().cpu().detach().numpy()
            pred_idx=(pred_score>0.5)
            
            #conflict type 1: good score conflict
            if (gt_id>0 and  pred_cls_id==0) or (gt_id==0 and pred_cls_id>0):
            # ng predicted as ok || ok predicted as ng
                dst_path = os.path.join(args.data,"conflict-"+tag,gt)
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                shutil.move(p,dst_path)
                shutil.move(p.replace(".png",".json"),dst_path)

            #conflict type2: good label come up with ng label
            elif pred_idx[0] and pred_idx[1:].any():
                dst_path = os.path.join(args.data,"conflict-"+tag,gt)
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                shutil.move(p,dst_path)
                shutil.move(p.replace(".png",".json"),dst_path)                

            #conflict type3: no label
            elif not pred_idx.any():
                dst_path = os.path.join(args.data,"conflict-"+tag,gt)
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                shutil.move(p,dst_path)
                shutil.move(p.replace(".png",".json"),dst_path)    


                
            if (not args.control.projectOperater.checking_train) and \
                (not args.control.projectOperater.checking_test):
                print("abort")
                return     
            #ui display progress
            args.control.projectOperater.dataCount += 1/len(image_paths)         

        print("done")


    def export_torchscript(self,args,save_path):
        print("export torchscript...")
        rand_input = torch.rand(1,3,args.input_h,args.input_w).cuda()

        from export.shufflenetv2 import shufflenet_v2_x1_0
        
        model = shufflenet_v2_x1_0(num_classes=args.classes)
        model_path=args.data.replace("params","models/checkpoint.pth.tar")
        checkpoint = torch.load(model_path) 
        static_dict = checkpoint['state_dict']
        model.load_state_dict(static_dict, strict=True)

        model.cuda()
        model.eval()
        
        torchscript = torch.jit.trace(model,rand_input,strict=False)
        localtime = time.localtime(time.time())
        date = "-".join([str(i) for i in localtime[0:3]])

        file_name="{}_multilabel_{}_{}x{}_{}.torchscript.pt".format(args.model_names,\
                                                                str(args.classes),\
                                                                str(args.input_w),\
                                                                str(args.input_h),\
                                                                date)
        torchscript.save(os.path.join(save_path,file_name))
        print("ok")

    def define_lr_scheduler(self, optimizer):
        steps_per_epoch=len(self.train_loader_)
        lr_scheduler = OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=steps_per_epoch, epochs=self.epoches,
                                        pct_start=0.2)
        return lr_scheduler