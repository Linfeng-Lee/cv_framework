from typing import Optional
from torchvision.transforms import *
from utils.util import load_yaml


class Transforms:
    def __init__(self, config: str):
        self.config = config

        self.args = {
            'train': [],
            'val': [],
            'test': [],
        }

        self._load_config()

    def _load_config(self):
        data = load_yaml(self.config)
        self.args['train'] = data.get('train')
        self.args['val'] = data.get('val')
        self.args['test'] = data.get('test')

    def get_trans(self) -> dict:

        # k,trans_args='train',train_trans_args
        for k, trans_args in self.args.items():
            _t = []

            # item=['Resize',[256,256]] or 'ToTensor'
            for item in trans_args:
                if isinstance(item, str):
                    _t.append(eval(item)())
                else:
                    if isinstance(item[0], str):
                        op = eval(item[0])
                        _t.append(op(*item[1:]))

            self.args[k] = transforms.Compose(_t)

        return self.args


if __name__ == '__main__':
    # val_transforms = transforms.Compose([
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    # ])
    # print(val_transforms)
    # print(type(val_transforms))

    t = Transforms(config='test_trans_args.yaml')
    trans = t.get_trans()
    print(trans)
