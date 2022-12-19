import json

import numpy as np
from random import shuffle

import torch
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class ExternalInputIterator:
    def __init__(self, file_path: str, batch_size: int, num_instances: int = 1, shuffled: bool = False):
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.shuffled = shuffled
        self.img_seq_length = num_instances

        with open(file_path) as f:
            images_dict = json.load(f)
        self.images_dict = images_dict
        self.list_of_pids = list(images_dict.key())
        self._num_classes = len(self.list_of_pids)
        self.all_indexs = list(range(len(self.list_of_pids)))
