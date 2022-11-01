import json
import os
import yaml

TIME_FORMAT = '%H:%M:%S'


#

def mkdir():
    ...


def load_yaml(yaml_path: str) -> dict:
    """
    :param yaml_path: xxx/xxx/xxx.yaml
    """
    assert yaml_path.endswith(".yaml") is True, f'file type must be yaml'
    assert os.path.exists(yaml_path) is True, f'{yaml_path}is error'
    with open(yaml_path, 'r', encoding='UTF-8') as f:
        conf = yaml.safe_load(f)
    return conf


def write_json(json_data, save_path: str):
    with open(save_path, "w") as f:
        json.dump(json_data, f)


def load_json(json_path: str):
    with open(json_path) as f:
        json_data = json.load(f)
    return json_data


def get_labels(path: str) -> list:
    labels = []
    if os.path.exists(path):
        labels = os.listdir(path)
    return labels


def load_aug_config(params: dict) -> list:
    use_aug_params = [k for k, v in params.items() if v]
    return use_aug_params


def get_balance_weight(beta, samples_per_cls, classes):
    import torch
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * classes

    weights = torch.tensor(weights).float()
    return weights


if __name__ == '__main__':
    # kw = {"1": True, "2": False, "3": True, "4": True}
    # a = load_aug_config(kw)
    # print(a)
    import numpy as np

    samples_per_cls = np.ones((10)) * 100
    samples_per_cls[0] = 90000
    print(samples_per_cls)
    period_weights = get_balance_weight(0.95, samples_per_cls=samples_per_cls,
                                        classes=10).cuda()
    print(period_weights)
