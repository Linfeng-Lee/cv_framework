import shutil, os, glob
import numpy as np
import torch
import cv2
import yaml
from loguru import logger


def display_config(data):
    for k, v in data.items():
        if k == 'augment':
            continue
        logger.info(f'[{k}]{type(v)} : {v}')

    logger.info('')

    for k, v in data['augment'].items():
        logger.info(f'[{k}]{type(v)} : {v}')


def path_join(*path):
    return os.path.join(*path)


def load_aug_config(params: dict) -> list:
    use_aug_params = [k for k, v in params.items() if v]
    return use_aug_params


def get_file_list(folder_path: str, p_postfix: list = None, sub_dir: bool = True) -> list:
    """
    获取所给文件目录里的指定后缀的文件,读取文件列表目前使用的是 os.walk 和 os.listdir ，这两个目前比 pathlib 快很多
    :param filder_path: 文件夹名称
    :param p_postfix: 文件后缀,如果为 [.*]将返回全部文件
    :param sub_dir: 是否搜索子文件夹
    :return: 获取到的指定类型的文件列表
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if p_postfix is None:
        p_postfix = ['.jpg']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = [x for x in glob.glob(folder_path + '/**/*.*', recursive=True) if
                 os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix]
    return file_list


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_balance_weight(beta, samples_per_cls, classes):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * classes

    weights = torch.tensor(weights).float()
    return weights


def vis_maps(img, predict, num_of_class, save_path):
    color_list = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                  [255, 0, 85], [255, 0, 170],
                  [0, 255, 0], [85, 255, 0], [170, 255, 0],
                  [0, 255, 85], [0, 255, 170],
                  [0, 0, 255], [85, 0, 255], [170, 0, 255],
                  [0, 85, 255], [0, 170, 255],
                  [255, 255, 0], [255, 255, 85], [255, 255, 170],
                  [255, 0, 255], [255, 85, 255], [255, 170, 255],
                  [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    img = np.array(img)
    vis_img = img.copy().astype(np.uint8)
    vis_predict = predict.copy().astype(np.uint8)
    vis_predict = cv2.resize(vis_predict, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    vis_predict_color = np.zeros((img.shape[0], img.shape[1], 3)) + 255
    for pi in range(0, num_of_class):
        index = np.where(vis_predict == pi)
        vis_predict_color[index[0], index[1], :] = color_list[pi]

    vis_predict_color = vis_predict_color.astype(np.uint8)
    vis_opencv = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    vis_addweight = cv2.addWeighted(vis_opencv, 0.4, vis_predict_color, 0.6, 0)
    cv2.imwrite(save_path[:-4] + ".png", vis_opencv)
    cv2.imwrite(save_path[:-4] + ".jpg", vis_addweight)


def accuracy_good_ng(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred_copy = pred.clone()
        pred_copy[pred_copy > 0] = 1
        target_copy = target.clone()
        target_copy[target_copy > 0] = 1
        correct = pred_copy.eq(target_copy.view(1, -1).expand_as(pred_copy))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, train_network_cls, filename='temp/checkpoint'):
    best_filename = "temp/model_best"
    if train_network_cls:
        filename = filename + "_landmark_.pth.tar"
        best_filename = best_filename + "_landmark_.pth.tar"
    else:
        filename = filename + "_cls_.pth.tar"
        best_filename = best_filename + "_cls_.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def make_dir(root_path, clear_flag=True):
    if os.path.exists(root_path):
        if clear_flag:
            shutil.rmtree(root_path)
            os.mkdir(root_path)
    else:
        os.mkdir(root_path)


def get_key(dct, value):
    return list(filter(lambda k: dct[k] == value, dct))


# accuracymultilabel
def accuracymultilabel(output, target):
    with torch.no_grad():
        num_instance, num_class = target.size()
        output[output > 0.5] = 1
        output[output <= 0.5] = 0

        count = 0
        gbcount = 0
        for i in range(num_instance):
            p = sum(np.logical_and(target[i].cpu().detach().numpy(), output[i].cpu().detach().numpy()))
            q = sum(np.logical_or(target[i].cpu().detach().numpy(), output[i].cpu().detach().numpy()))
            count += p / q

            if output[i][0] == target[i][0]:
                gbcount += 1

    return [gbcount / num_instance], [count / num_instance]


def load_yaml(yaml_path: str) -> dict:
    """
    :param yaml_path: xxx/xxx/xxx.yaml
    """
    assert yaml_path.endswith(".yaml") is True, f'file type must be yaml'
    assert os.path.exists(yaml_path) is True, f'{yaml_path}is error'
    with open(yaml_path, 'r', encoding='UTF-8') as f:
        conf = yaml.safe_load(f)
    return conf