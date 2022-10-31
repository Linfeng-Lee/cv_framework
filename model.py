import os
from loguru import logger
from utils.common import load_yaml


class Model:
    def __init__(self, config: str):
        self.net_config = config
        logger.info(f'Input net_config path: {self.net_config}.')

        self.project = None
        self.net = None
        self.net_type = None
        self.classes = None
        self.pretrain = None
        self.test = None
        self.train = None
        self.image_w = None
        self.image_h = None
        self.batch_size = None
        self.start_epoch = None
        self.epochs = None
        self.resume = None
        self.device = None
        self.worker = None
        self.seed = None
        self.lr = None
        self.loss = None
        self.threshold = None
        self.augment = None

        self.load_net_config()

        self.crop = None
        self.change_colorspace = None
        self.contrast_normalization = None
        self.gaussian_noise = None
        self.multiply = None
        self.scale = None
        self.rotate = None
        self.shear = None
        self.grayscale = None
        self.hue_and_saturation = None
        self.flip_left_right = None
        self.flip_up_down = None

        self.load_aug_config()

        self.trainer = self._trainer()

    def load_net_config(self):
        net_conf = load_yaml(self.net_config)
        logger.info(f'net_config :{net_conf}')

        self.project = net_conf['project']
        self.net = net_conf['net']
        self.net_type = net_conf['net_type']
        self.classes = net_conf['classes']
        self.pretrain: bool = net_conf['pretrain']
        self.test = net_conf['test']
        self.train = net_conf['train']
        self.image_w: int = net_conf['image_w']
        self.image_h: int = net_conf['image_h']
        self.batch_size = net_conf['batch_size']
        self.start_epoch = net_conf['start_epoch']
        self.epochs = net_conf['epochs']
        self.resume: str = net_conf['resume']
        self.device = net_conf['device']
        self.worker: int = net_conf['worker']
        self.seed = net_conf['seed']
        self.lr = net_conf['lr']
        self.loss = net_conf['loss']
        self.threshold: float = net_conf['threshold']
        self.augment = net_conf['augment']

    def load_aug_config(self):
        aug_config = load_yaml(os.path.join('project', self.augment))
        logger.info(f'augment_config:{aug_config}')
        self.crop = aug_config['crop']
        self.change_colorspace = aug_config['change_colorspace']
        self.contrast_normalization = aug_config['contrast_normalization']
        self.gaussian_noise = aug_config['gaussian_noise']
        self.multiply = aug_config['multiply']
        self.scale = aug_config['scale']
        self.rotate = aug_config['rotate']
        self.shear = aug_config['shear']
        self.grayscale = aug_config['grayscale']
        self.hue_and_saturation = aug_config['hue_and_saturation']
        self.flip_left_right = aug_config['flip_left_right']
        self.flip_up_down = aug_config['flip_up_down']

    def _trainer(self):
        if self.net_type == "classification":
            from trainer.base_trainer import BaseTrainer
            return BaseTrainer()

        elif self.net_type == "segmentation":
            from trainer.segmentation_trainer import SegmantationTrainer
            return SegmantationTrainer()

        elif self.net_type == "multitask":
            raise NotImplementedError

        elif self.net_type == "multilabel-classification":
            from trainer.multi_label_classify_trainer import MultiLabelClassifyTrainer
            return MultiLabelClassifyTrainer()

        elif self.net_type == "embedding-classification":
            from trainer.embedding_trainer import EmbeddingTrainer
            return EmbeddingTrainer()

        else:
            raise NotImplementedError


if __name__ == '__main__':
    model = Model('project/demo/net_config.yaml')
