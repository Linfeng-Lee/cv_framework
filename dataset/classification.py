import os, os.path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from dataset.folder_base import VisionDatasetBase
from dataset.folder_base import IMG_EXTENSIONS
from dataset.base import BaseDataset


class BaseDatasetClassify(BaseDataset):
    def __init__(self,
                 root: str,
                 loader: Callable,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 extensions: Optional[Tuple[str, ...]] = None
                 ):
        super(BaseDatasetClassify, self).__init__()
        self.classes = self._get_classes(root)
        self.class_to_idx = self._classes2idx(self.classes)

        if loader is None:
            self.loader = self.default_loader
        else:
            self.loader = loader

        self.transform = transform
        self.target_transform = target_transform

        if extensions is not None:
            # self.extensions = extensions
            self.IMG_EXTENSIONS = extensions

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @staticmethod
    def load_class_id_map(class_id_map_path: str) -> dict:

        if not os.path.exists(class_id_map_path):
            return dict()

        class_id_map = dict()
        with open(class_id_map_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                # k=0,1,2,...
                # v=classes1,classes2,...
                k, v = line.split("x")
                # class_id_map={0:"classes1",1:"classes2",...}
                class_id_map[k] = v
        return class_id_map

    def save_class_id_map(self, save_path: str) -> None:
        with open(save_path, 'w') as f:
            # classes_to_idx=classes1:0,classes1:1,...
            # classes_id_map = 0xclasses1,1xclasses2,...
            f.writelines(['{}x{}\n'.format(v, k) for k, v in self.class_to_idx.items()])

    @staticmethod
    def _get_classes(path: str) -> list:
        # classes=[classes1,classes2,...]
        classes = [d.name for d in os.scandir(path) if d.is_dir()]
        classes.sort()
        return classes

    @staticmethod
    def _classes2idx(classes: list) -> dict:
        # class_to_idx={classes1:0,classes:1,...}
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx


class ImplDatasetClassification(BaseDatasetClassify):
    def __init__(self,
                 root: str,
                 loader: Callable,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 extensions: Optional[Tuple[str, ...]] = None
                 ):
        super(ImplDatasetClassification, self).__init__(root=root,
                                                        loader=loader,
                                                        transform=transform,
                                                        target_transform=target_transform,
                                                        extensions=extensions
                                                        )

        self.samples = self.get_samples(root, class_to_idx=self.class_to_idx)

        if len(self.samples) == 0:
            msg = f"Found 0 files in subfolders of: {self.root}\n"
            raise RuntimeError(msg)
        if extensions is not None:
            print(f"Supported extensions are: {extensions}")

        self.targets = [item[1] for item in self.samples]

    def __getitem__(self, index: int) -> Any:
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def get_samples(self, path: str, **kwargs):
        class_to_idx = kwargs.get('class_to_idx')

        samples = []
        path = os.path.expanduser(path)

        # target:classes0,classes1,...(low->high)
        for target in sorted(class_to_idx.keys()):
            target_path = os.path.join(path, target)
            if not os.path.exists(target_path):
                continue

            for root, dirs, files in os.walk(target_path):
                for file in files:
                    if self.is_valid_image(file):
                        file = os.path.join(root, file)
                        item = (file, class_to_idx[target])
                        # samples=[('xxx/xxx.jpg',1),(xxx/xxx.jpg,0),...]
                        samples.append(item)
        random.shuffle(samples)
        return samples


class DatasetClassification(ImplDatasetClassification):
    def __init__(self,
                 root: str,
                 loader: Callable,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 extensions: Optional[Tuple[str, ...]] = None
                 ):
        super(DatasetClassification, self).__init__(root=root,
                                                    loader=loader,
                                                    transform=transform,
                                                    target_transform=target_transform,
                                                    extensions=extensions)
