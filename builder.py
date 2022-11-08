from loguru import logger

from trainer.base_trainer import BaseTrainer
from trainer.segmentation_trainer import SegmantationTrainer
from trainer.embedding_trainer import EmbeddingTrainer
from trainer.multi_label_classify_trainer import MultiLabelClassifyTrainer
from trainer.multi_task_trainer import MultiTaskTrainer

_Type = {
    'basetrainer': BaseTrainer,
    'segmentation': SegmantationTrainer,
    'embedding_classification': EmbeddingTrainer,
    'multilabel_classification': MultiLabelClassifyTrainer,
    'multitask': MultiTaskTrainer,
}


def build_model(network_type: str, args):
    logger.info(f'Input Network Type: {network_type}')
    if network_type not in _Type.keys():
        logger.error(f'Do not find {network_type} network -> exit')
    trainer = _Type[network_type]
    trainer.init_trainer(args)


if __name__ == '__main__':
    build_model('embedding_classification', {})
