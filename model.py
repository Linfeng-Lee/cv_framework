import os
import platform
from typing import Union, List, Tuple, Callable
import numpy as np
from loguru import logger
from torchvision import transforms

from augment.aug_transforms import AugTransform
from augment.transforms_keypoints import AugKeypoints
from augment.aug_config import config_aug
from utils.util import get_balance_weight, auto_mkdir_project, load_aug_config, display_config

from trainer.multi_label_classify import MultiLabelClassifyTrainer
from trainer.embedding import EmbeddingTrainer
from trainer.classification import ClassificationTrainer
from trainer.multi_task import MultiTaskTrainer
from trainer.segmentation import SegmentationTrainer

_Network_ = {
    'classification': ClassificationTrainer,
    'embedding': EmbeddingTrainer,
    'multi_label_classify': MultiLabelClassifyTrainer,
    'segmentation': SegmentationTrainer,
    'multi_task': MultiTaskTrainer,
}


class Model:
    def __init__(self, args=None):
        self.args = args
        self.ROOT = 'project'
        self.trainer = None
        self._init()

    def _init(self):
        auto_mkdir_project(self.args.project)
        self._update_args()
        self._init_transforms()
        self._init_trainer()

    def _update_args(self):
        logger.info('🚀Update args....')
        self.args.asymmetry_id = False

        samples_per_cls = np.ones(self.args.classes) * 100
        samples_per_cls[0] = 90000

        # yacs do not match np type or cuda type
        self.period_weights = self.args.period_weights
        self.period_weights = get_balance_weight(0.95, samples_per_cls=samples_per_cls,
                                                 classes=self.args.classes)  # .cuda()
        self.args.period_n_min = 1 * self.args.input_h * self.args.input_w
        self.args.period_thresh = 0.9

        # check class weight
        if self.args.class_weight:
            assert len(self.args.class_weight) == self.args.classes

        # check platform
        if platform.system().lower() == "windows":
            self.args.worker = 0

        display_config(self.args)

        logger.success('Update args done.\n')

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

    def _init_trainer(self):

        self.train_data_path = os.path.join(str(self.args.data_root), "train")
        self.val_data_path = os.path.join(str(self.args.data_root), "test")
        assert os.path.exists(self.train_data_path)
        assert os.path.exists(self.val_data_path)
        self.args.train_data_path = self.train_data_path
        self.args.val_data_path = self.val_data_path

        self.tensorboard_save_path = os.path.join(self.ROOT, self.args.project, "runs")

        if self.args.task_type == "classification":
            from trainer.base_trainer import BaseTrainer
            self.trainer = BaseTrainer()
            self.trainer.init_trainer_v2(self.args,
                                         self.train_transform,
                                         self.val_transform,
                                         self.tensorboard_save_path)

        elif self.args.task_type == "segmentation":
            from trainer.segmentation_trainer import SegmantationTrainer
            self.trainer = SegmantationTrainer()
            self.trainer.init_trainer_v2(self.args,
                                         self.tensor_transforms,
                                         self.keypoint_transforms,
                                         self.tensorboard_save_path)

        elif self.args.task_type == "multilabel-classification":
            from trainer.multi_label_classify_trainer import MultiLabelClassifyTrainer
            self.trainer = MultiLabelClassifyTrainer()

            self.trainer.init_trainer_v2(self.args,
                                         self.train_transform,
                                         self.tensor_transforms,
                                         self.tensorboard_save_path)

        elif self.args.task_type == "embedding-classification":
            from trainer.embedding_trainer import EmbeddingTrainer
            self.trainer = EmbeddingTrainer()

            self.trainer.init_trainer_v2(self.args,
                                         self.train_transform,
                                         self.val_transform,
                                         self.tensorboard_save_path)


        elif self.args.task_type == "multitask":
            network = _Network_[self.args.task_type]
            self.trainer = network()
        else:
            raise NotImplementedError

    def train(self):
        self.args.present_net = self.args.net
        self.args.training = True
        self.args.automate = True
        self.args.dataCount = 0
        self.args.loss = 0
        self.args.trainAuc = 0
        save_path = os.path.join(self.ROOT, self.args.project, 'models')
        logger.info(f'🚀 {self.args.net} start train.')

        self.trainer.resume(self.args.resume, strict=True)

        self.trainer.fit(save_path=save_path,
                         normalize_transpose=self.normalize_transpose)

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
        self.trainer.test(self.args)
        logger.info(f'{self.args.net} end test.')

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
    #     self.trainer.resume(self.dataRoot.replace("config", "models/checkpoint.pth.tar"), True)
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
    #     self.trainer.resume(self.dataRoot.replace("config", "models/checkpoint.pth.tar"), True)
    #     self.trainer.check_data_conflict(self.args, tensor_transforms, "test")

    def splitUnknown(self, modelNameList, ):
        pass


if __name__ == '__main__':
    def add(a, b):
        return a + b


    def fun(data: Union[int, list, tuple], b: int, f: Callable):
        print(type(data))
        print(type(f))
        a = f(data, b)
        print(a)


    fun(1, 2, add)
