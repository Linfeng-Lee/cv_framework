import os, sys, random
from typing import Tuple, Any

import numpy as np
from torch.utils.data.sampler import BatchSampler
from PIL import Image

from torchvision.datasets.vision import VisionDataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def get_nums_of_class(dir):
    '''
    用来获取训练目录下的每个类别的样本扥数据量
    :param dir: 训练样本目录
    :return:
    '''
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    if sys.version_info >= (3, 5):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    def is_valid_file(x):
        return x.lower().endswith(IMG_EXTENSIONS)

    num_of_class = dict()
    for target in sorted(class_to_idx.keys()):
        nums = 0
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for dirPath, dirNames, fileNames in os.walk(d):
            for file in fileNames:
                if file.split('.')[-1].lower() in {"png", 'bmp', 'jpg', 'jpeg'}:
                    nums += 1
        num_of_class[target] = nums

    return num_of_class


def save_class_id_map(class_id_map_save_path, ds):
    '''
    保存深度网络的训练id
    :param conf:
    :param ds:
    :return:
    '''
    with open(class_id_map_save_path, 'w') as f:
        # classes_to_idx=classes1:0,classes1:1,...
        # classes_id_map = 0xclasses1,1xclasses2,...
        f.writelines(['{}x{}\n'.format(v, k) for k, v in ds.class_to_idx.items()])


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        # 这里指的是对应lebel已经使用的数据个数
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                # 当数据的使用数量大于这个lebel具有的数据长度的时候，重置已使用数据的数量
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class VisionDatasetBase(VisionDataset):

    def load_class_id_map(self, conf) -> dict:
        """
        加载网络输出id的映射
        :param conf:
        :return:
        """
        if not os.path.exists(conf.class_id_map_path):
            return dict()

        class_id_map = dict()
        with open(conf.class_id_map_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                # k=0,1,2,...s
                # v=classes1,classes2,...
                k, v = line.split("x")
                # class_id_map={0:"classes1",1:"classes2",...}
                class_id_map[k] = v
        return class_id_map

    def has_file_allowed_extension(self, filename: str, extensions: Tuple[str, ...]) -> bool:
        """Checks if a file is an allowed extension.

        Args:
            filename (string): path to a file
            extensions (tuple of strings): extensions to consider (lowercase)

        Returns:
            bool: True if the filename ends with one of given extensions
        """
        return filename.lower().endswith(extensions)

    def is_image_file(self, filename: str) -> bool:
        """Checks if a file is an allowed image extension.

        Args:
            filename (string): path to a file

        Returns:
            bool: True if the filename ends with a known image extension
        """
        return self.has_file_allowed_extension(filename, IMG_EXTENSIONS)

    def make_dataset_base(self,
                          dir,
                          class_to_idx: dict,
                          extensions=None,
                          is_valid_file=None,
                          nums_of_class=None):
        images = []
        dir = os.path.expanduser(dir)
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x):
                return self.has_file_allowed_extension(x, extensions)

        # target:classes0,classes1,...(low->high)
        for target in sorted(class_to_idx.keys()):

            # dir=/home/user/data/xxx
            # d=/home/user/data/xxx/classes1
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            index = 0
            for dirPath, dirNames, fileNames in os.walk(d):
                for file in fileNames:
                    if file.split('.')[-1].lower() in {"png", 'bmp', 'jpg', 'jpeg'}:
                        file = os.path.join(dirPath, file)
                        item = (file, class_to_idx[target])
                        # images=[('xxx/xxx.jpg',1),(xxx/xxx.jpg,0),...]
                        images.append(item)

        random.shuffle(images)
        return images

    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    # TODO: specify the return type
    def accimage_loader(self, path: str) -> Any:
        import accimage
        try:
            return accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return self.pil_loader(path)

    def default_loader(self, path: str) -> Any:
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return self.accimage_loader(path)
        else:
            return self.pil_loader(path)


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5
