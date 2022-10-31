
from random import shuffle
import torch
from torch.utils.data import DataLoader

from dataset.folder_mask import ImageFolder as trainImageFolder
from dataset.folder_classify import ImageFolder as TrainFolderClassify
from dataset.folder_base import get_nums_of_class
from dataset.folder_base import save_class_id_map
from dataset.folder_base import BalancedBatchSampler
from dataset.folder_multilabelclassify import ImageFolder as TrainFolderMultiClassify
from torch.utils.data import WeightedRandomSampler

def get_dataset(imgs_folder, train_transform, target_transform, nums_of_class, remove_no_json_sample_flag=False,
                convert_float_flag=False, return_path=False,**kwargs):
    ds = trainImageFolder(imgs_folder, train_transform, target_transform, nums_of_class=nums_of_class,
                          remove_no_json_sample_flag=remove_no_json_sample_flag, convert_float_flag=convert_float_flag,
                          return_path=return_path,**kwargs)
    class_num = len(ds.classes)
    return ds, class_num


def get_loader_mask(root_path, tensor_transform, mask_transform,
                    balance_n_classes, balance_n_samples,
                    class_id_map_save_path, remove_no_json_sample_flag=False,
                    convert_float_flag=False, return_path=False,**kwargs):
    """
    数据集读取处理函数方法
    :return: 返回数据加载器，跟总的类别数量

    Args:
        balance_n_classes:
    """
    nums_of_class = get_nums_of_class(root_path)
    ds, class_num = get_dataset(root_path, tensor_transform, target_transform=mask_transform,
                                nums_of_class=nums_of_class,
                                remove_no_json_sample_flag=remove_no_json_sample_flag,
                                convert_float_flag=convert_float_flag,
                                return_path=return_path,**kwargs)
    save_class_id_map(class_id_map_save_path, ds)
    # 根据样本平衡方法进行数据集的读取处理
    # batch_sample = BalancedBatchSampler(torch.tensor(ds.targets), n_classes=balance_n_classes,
    #                                           n_samples=balance_n_samples)
    # dataloader = DataLoader(ds, batch_sampler=batch_sample)
    dataloader = DataLoader(ds,batch_size=balance_n_samples*balance_n_classes,shuffle=True)

    return dataloader


def get_loader_class(root_path, transform, balance_n_classes, balance_n_samples, class_id_map_save_path):
    """
    数据集读取处理函数方法
    :param conf:
    :return: 返回数据加载器，跟总的类别数量
    """
    ds = TrainFolderClassify(root_path, transform)
    save_class_id_map(class_id_map_save_path, ds)
    # 根据样本平衡方法进行数据集的读取处理
    batch_sample = BalancedBatchSampler(torch.tensor(ds.targets), n_classes=balance_n_classes,
                                              n_samples=balance_n_samples)
    dataloader = DataLoader(ds, batch_sampler=batch_sample)
    return dataloader


def get_loader_multi_label_class(root_path, transform, balance_n_classes, balance_n_samples, class_id_map_save_path, valtransform, weights=None, is_training=True):
    """
    数据集读取处理函数方法
    :param conf:
    :return: 返回数据加载器，跟总的类别数量
    """
    if is_training:
        ds = TrainFolderMultiClassify(root_path, transform, is_valid_file=False)
        save_class_id_map(class_id_map_save_path, ds)

        if weights is None or len(ds.classes) == weights.sum():
            dataloader = DataLoader(ds, batch_size=balance_n_classes * balance_n_samples, shuffle=True)
        else:
            sampler = WeightedRandomSampler(weights, num_samples=len(ds.samples), replacement=True)
            dataloader = DataLoader(ds, batch_size=balance_n_classes*balance_n_samples, sampler=sampler)

    else:
        # val
        ds = TrainFolderMultiClassify(root_path, valtransform, is_valid_file=True)
        dataloader = DataLoader(ds, batch_size=balance_n_classes*balance_n_samples, shuffle=True)


    return dataloader

