from model import Model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training yaml config.')
    parser.add_argument('--yaml', type=str, default='params/shuangjing.yaml', help='input your training yaml file.')
    args = parser.parse_args()

    model = Model(args.yaml)
    model.run()
