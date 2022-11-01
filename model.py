import os
import numpy as np
import platform

from loguru import logger
from easydict import EasyDict as edict
from torchvision import transforms

from cv_framework.aug_config import config_aug
from augment.transforms import AugTransform
from augment.transforms_keypoints import AugKeypoints
from utils.util import get_balance_weight, load_yaml, load_aug_config, path_join, display_config


class Model:
    def __init__(self, config=None):
        if config is None:
            return

        if isinstance(config, str):
            self.config = load_yaml(config)
            self.args = edict(self.config)

        self.ROOT = 'project'

        self.trainer = None

        self._init()

    def _update_args(self):
        logger.info('🚀Update args....')
        self.args.asymmetry_id = False

        samples_per_cls = np.ones(self.args.classes) * 100
        samples_per_cls[0] = 90000
        self.args.period_weights = get_balance_weight(0.95, samples_per_cls=samples_per_cls,
                                                      classes=self.args.classes).cuda()
        self.args.period_n_min = 1 * self.args.input_h * self.args.input_w
        self.args.period_thresh = 0.9

        # check class weight
        if self.args.class_weight:
            assert len(self.args.class_weight) == self.args.classes

        # check platform
        if platform.system().lower() == "windows":
            self.args.worker = 0

        display_config(self.args)

        logger.success('Updata args done.\n')

    def _init_transforms(self):
        logger.info('🚀Init transforms...')
        augment_set = load_aug_config(self.args.augment)
        iaa_aug_seq, self.normalize, self.normalize_transpose = config_aug(self.args.input_h, self.args.input_w,
                                                                           augment_set)

        self.train_transform = AugTransform(iaa_aug_seq, self.normalize)
        self.val_transform = transforms.Compose([transforms.Resize((self.args.input_h, self.args.input_w)),
                                                 transforms.ToTensor(), self.normalize])

        self.keypoint_transforms = AugKeypoints(p=1, seq_det=iaa_aug_seq, convert_float_coord=True)

        self.tensor_transforms = transforms.Compose([transforms.ToTensor(), self.normalize, ])

        logger.success('Init transforms done.\n')

    def _init(self):
        self._init_transforms()
        self._init_trainer()
        self._update_args()

    def _init_trainer(self):

        self.train_data_path = path_join(self.args.data_root, "train")
        self.val_data_path = path_join(self.args.data_root, "test")

        if self.args.task_type == "classification":
            from tool.trainer.base_trainer import BaseTrainer
            self.trainer = BaseTrainer()
            self.trainer.init_trainer(self.net,
                                      self.args.lr,
                                      self.train_data_path,
                                      self.val_data_path,
                                      self.train_transform,
                                      self.val_transform,
                                      self.args.gpus,
                                      self.args.classes,
                                      self.args.batch_size,
                                      self.worker,
                                      tensorboard_save_path=path_join(self.ROOT, "params", "models/tensorboard"),
                                      criterion_list=self.args.criterion_list,
                                      pretrained=False,
                                      args=self.args)

        elif self.args.task_type == "segmentation":
            from tool.trainer.segmentation_trainer import SegmantationTrainer
            self.trainer = SegmantationTrainer()
            self.trainer.init_trainer(self.args.net,
                                      self.args.lr,
                                      self.train_data_path,
                                      self.val_data_path,
                                      self.args.gpus,
                                      self.args.classes,
                                      classify_transform=self.train_transform,
                                      criterion_list=self.args.criterion_list,
                                      tensor_transform=self.tensor_transforms,
                                      mask_transform=self.keypoint_transforms,
                                      balance_n_classes=self.args.balance_n_classes,
                                      balance_n_samples=self.args.balance_n_samples,
                                      class_id_map_save_path=self.args.class_id_map_save_path,
                                      remove_no_json_sample_flag=False,
                                      convert_float_flag=self.args.convert_float_flag,
                                      return_path=True,
                                      period_thresh=self.args.period_thresh,
                                      period_n_min=self.args.period_n_min,
                                      period_weights=self.args.period_weights,
                                      asymmetry_id=False,
                                      weight=self.args.class_weight,
                                      args=self.args)

        elif self.args.task_type == "multilabel-classification":
            from tool.trainer.multi_label_classify_trainer import MultiLabelClassifyTrainer
            self.trainer = MultiLabelClassifyTrainer()
            self.trainer.init_trainer(self.net,
                                      self.args.lr,
                                      self.train_data_path,
                                      self.val_data_path,
                                      self.args.gpus,
                                      self.args.classes,
                                      classify_transform=self.train_transform,
                                      criterion_list=["AsymmetricLoss"],
                                      tensor_transform=self.tensor_transforms,
                                      balance_n_classes=self.args.balance_n_classes,
                                      balance_n_samples=self.args.balance_n_samples,
                                      class_id_map_save_path=self.args.class_id_map_save_path,
                                      remove_no_json_sample_flag=False,
                                      convert_float_flag=self.args.convert_float_flag,
                                      return_path=True,
                                      period_n_min=self.args.period_n_min,
                                      weights=None,
                                      pos_weights=None,
                                      asymmetry_id=False,
                                      mask_classes=self.args.classes,
                                      epoches=self.args.epochs,
                                      args=self.args)

        elif self.args.task_type == "embedding-classification":
            from tool.trainer.embedding_trainer import EmbeddingTrainer
            self.trainer = EmbeddingTrainer()
            self.trainer.init_trainer(self.args.net,
                                      self.args.lr,
                                      self.train_data_path,
                                      self.val_data_path,
                                      self.train_transform,
                                      self.val_transform,
                                      self.args.gpus,
                                      self.args.classes,
                                      self.args.batch_size,
                                      self.args.worker,
                                      tensorboard_save_path=path_join(self.ROOT, "params", "models/tensorboard"),
                                      criterion_list=self.args.criterion_list,
                                      pretrained=True,
                                      args=self.args)

        elif self.args.task_type == "multitask":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def train(self):
        self.args.present_net = self.args.net
        self.args.training = True
        self.args.automate = True
        self.args.dataCount = 0
        self.args.loss = 0
        self.args.trainAuc = 0

        logger.info(f'🚀 {self.args.net} start train.')

        self.trainer.resume(self.args.resume, strict=True)

        self.trainer.fit(start_epoch=self.args.start_epoch,
                         epochs=self.args.epochs,
                         top_k=2,
                         save_path=path_join(self.ROOT, self.args.project, 'models'),
                         normalize_transpose=self.normalize_transpose,
                         args=self.args)

        logger.info(f'{self.args.net} end train.')

    def test(self):

        self.args.present_net = self.args.net
        self.args.testing = True
        self.args.automate = True
        self.args.dataCount = 0
        self.args.loss = 0
        self.args.trainAuc = 0

        logger.info(f'{self.args.net} end test.')
        self.trainer.resume(self.args.resume, True)
        self.trainer.validate(0, 1, self.args)
        logger.info(f'{self.args.net} end test.')

    # def export_model(self):
    #     export_path = path_join(self.ROOT, self.args.project, 'export')
    #
    #     self.trainer.export_torchscript(self.args, export_path)
    #     if self.args.task_type == "segmentation":
    #         self.trainer.export_onnx(self.args, export_path)

    # def checkTrainDataConflict(self):
    #     self.control.projectOperater.dataCount = 0
    #     self.control.projectOperater.trainAcu = 0
    #     self.control.projectOperater.testAcu = 0
    #     self.control.projectOperater.loss = 0
    #     self._init()
    #     self.dataRoot = self.control.projectManager._getDataRoot()
    #     tensor_transforms = transforms.Compose([transforms.Resize((self.imageSize[1], self.imageSize[0])),
    #                                             transforms.ToTensor(),
    #                                             self.normalize
    #                                             ])
    #
    #     self.trainer.resume(self.dataRoot.replace("params", "models/checkpoint.pth.tar"), True)
    #     self.trainer.check_data_conflict(self.args, tensor_transforms, "train")
    #
    # # network parser

    #
    # def checkTestDataConflict(self):
    #     self.control.projectOperater.dataCount = 0
    #     self.control.projectOperater.trainAcu = 0
    #     self.control.projectOperater.testAcu = 0
    #     self.control.projectOperater.loss = 0
    #     self._init()
    #     self.dataRoot = self.control.projectManager._getDataRoot()
    #
    #     tensor_transforms = transforms.Compose([transforms.Resize((self.imageSize[1], self.imageSize[0])),
    #                                             transforms.ToTensor(),
    #                                             self.normalize
    #                                             ])
    #
    #     self.trainer.resume(self.dataRoot.replace("params", "models/checkpoint.pth.tar"), True)
    #     self.trainer.check_data_conflict(self.args, tensor_transforms, "test")

    def splitUnknown(self, modelNameList, ):
        pass

    def run(self):
        self.train()
        logger.success('model启动成功!')
