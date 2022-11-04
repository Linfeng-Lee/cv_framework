import argparse
from config.default import merge_from_file
from test.classification import test as cls_test
from test.segmentation import test as seg_test
from test.multilabel_classification import test as mul_cls_test
from test.embedding_classification import test as meb_cls_test
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training yaml config.')
    parser.add_argument('--yaml', type=str, default='config/shuangjing_emb_cls.yaml',
                        help='input your training yaml file.')
    parser.add_argument('--weight', type=str, default='',
                        help='test weight')

    args = parser.parse_args()

    config = merge_from_file(args.yaml)
    if args.weight != '':
        config.resume = args.weight

    test_func = {
        'classification': cls_test,
        'multilabel-classification': mul_cls_test,
        'segmentation': seg_test,
        'embedding-classification': meb_cls_test,
    }
    test = test_func[config.task_type](config)
