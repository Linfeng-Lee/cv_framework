import time
import argparse
from loguru import logger
from loguru import logger
from tqdm import tqdm
from model import Model

from config.default import merge_from_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training yaml config.')
    parser.add_argument('--yaml', type=str, default='config/shuangjing_emb_cls.yaml',
                        help='input your training yaml file.')
    parser.add_argument('--test', default=False, action='store_true',
                        help='use this flag to test data')
    parser.add_argument('--weight', type=str, default='',
                        help='test weight')
    args = parser.parse_args()

    config = merge_from_file(args.yaml)

    if not args.test:
        model = Model(config)
        model.train()
    else:
        if args.weight == '':
            logger.warning(f'input test weight is empty,using train checkpoint to test.')
        else:
            config.resume = args.weight
        model = Model(config)
        model.test()
