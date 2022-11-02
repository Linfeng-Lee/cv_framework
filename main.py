import numpy as np

from model import Model
import argparse
from yacs.config import CfgNode as CN
from config.default import get_default_config, merge_from_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training yaml config.')
    parser.add_argument('--yaml', type=str, default='config/shuangjing_emb_cls.yaml',
                        help='input your training yaml file.')
    args = parser.parse_args()
    config = merge_from_file(args.yaml)
    model = Model(config)
    model.run()
