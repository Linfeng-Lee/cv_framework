import argparse
import time

from loguru import logger
from tqdm import tqdm
from model import Model

from config.default import merge_from_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training yaml config.')
    parser.add_argument('--yaml', type=str,
                        default='/home/lee/PycharmProjects/cv_framework/config/shuangjing_emb_cls.yaml',
                        help='input your training yaml file.')
    args = parser.parse_args()
    config = merge_from_file(args.yaml)
    model = Model(config)
    model.run()