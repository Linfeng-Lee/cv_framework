import time
import argparse
from loguru import logger
from loguru import logger
from tqdm import tqdm
from model import Model

from config.default import merge_from_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training yaml config.')
    parser.add_argument('--yaml', type=str,
                        default='config/deyuan_short_yellow_cls.yaml',
                        help='input your training yaml file.')
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--weight', type=str, default='', help='test weight')
    args = parser.parse_args()

    config = merge_from_file(args.yaml)
    model = Model(config)
    if args.train:
        model.train()
    # elif args.test:
    #     if args.weight == '':
    #         logger.warning(f'please input you test weight.')
    #         exit()
    #     else:
    #         logger.info(f'test weight:{args.weight}')
    #         model.test()
