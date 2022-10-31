import json
import os
import yaml


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
