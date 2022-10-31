import sys,os,os.path,json,random,pathlib
import numpy as np
import datetime
import cv2
from tqdm import tqdm
from typing import Any, Callable,Dict, List, Optional, Tuple
from augment import transforms as aug_transforms

from dataset.folder_base import VisionDatasetBase
from dataset.folder_base import IMG_EXTENSIONS

class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w',encoding="utf-8") as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)

class DatasetFolder(VisionDatasetBase):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/[...]/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/[...]/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        self.is_valid_file = is_valid_file
        if is_valid_file:
            self.reload_data_path = os.path.join(root,"test.json")
        else:
            self.reload_data_path = os.path.join(root,"train.json")

        self.samples, self.classes, self.class_to_count, self.class_to_idx = self.load_data(self.root)

        if loader is None:
            self.loader=self.default_loader
        else:
            self.loader = loader

        self.extensions = extensions

        self.save_aug_img_count = 0
        self.aug_data_save_path = "temp/aug/"

    def load_data(self, data_path: str) -> list:
        """
        根据ignore里的text进行去除相应的字符信息。
        """
        data_list = self.get_data_list(data_path)
        t_data_list = []
        classes = []
        class_to_count = {}
        class_to_idx = {}

        for img_path in tqdm(data_list):
            labels_json_path = os.path.splitext(img_path)[0] + ".json"
            if not os.path.exists(labels_json_path):
                print('there is no json in {}'.format(img_path))
                continue

            labels_json = self.keypoint_loader(labels_json_path)

            labelsdic = {}
            for label, val in labels_json.get("flags").items():
                if label == "__ignore__":
                    continue
                label_digit = self.get_label_head_digit(label)
                if label_digit == -1:
                    print("label头部沒有数字！path：{}".format(img_path))

                class_to_idx.setdefault(label_digit,label)
                class_to_count.setdefault(label_digit,0)

                if val:
                    class_to_count[label_digit] += 1
                    labelsdic[str(label_digit)] = label

            if not len(labelsdic):
                print('there is no label in {}'.format(img_path))
                continue

            item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem, 'label': labelsdic}
            t_data_list.append(item)
            classes.extend(list(labelsdic.keys()))
            classes = list(set(classes))

        class_to_idx = dict(sorted(class_to_idx.items(), key=lambda x:x[0]))
        class_to_idx = dict(zip(class_to_idx.values(),class_to_idx.keys()))
        classes=list(class_to_idx.keys())

        return t_data_list, classes, class_to_count, class_to_idx

    def get_data_list(self,dir, extensions=None, is_valid_file=None):
        images = []
        dir = os.path.expanduser(dir)

        if not os.path.isdir(dir):
            return None
        for dirPath, dirNames, fileNames in os.walk(dir):
            for file in fileNames:
                if file.split('.')[-1].lower() in {"png", 'bmp', 'jpg', 'jpeg'}:
                    file = os.path.join(dirPath, file)
                    images.append(file)
        random.shuffle(images)
        return images

    def get_label_head_digit(self, label):

        digitStr=str(label.split("_")[0])

        return digitStr


    def _find_classes(self, dir, is_valid_file=None):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def keypoint_loader(self,path):
        with open(path,'r',encoding="utf-8")as f:
            keypoints_json=json.load(f)
        return keypoints_json

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        cur_info = self.samples[index]
        sample = self.loader(cur_info["img_path"])

        target = len(self.classes)*[0]
        for label in cur_info["label"]:
            target[int(label)] = 1

        if self.transform is not None:
            sample = self.transform(sample)

        if self.save_aug_img_count < 100 and not self.is_valid_file:
            aug_path = self.aug_data_save_path + os.path.basename(cur_info["img_path"])[:-4] + "_aug.png"
            norm = self.transform.transforms_.transforms[1]
            normalize_transpose = aug_transforms.NormalizeTranspose(mean=norm.mean, std=norm.std)
            img_aug = normalize_transpose(sample)
            img_aug = np.array(img_aug.cpu().detach().numpy(), dtype=np.uint8)
            img_aug = cv2.cvtColor(img_aug.transpose(1,2,0), cv2.COLOR_RGB2BGR)
            # cv2.imwrite(aug_path, img_aug)
            self.save_aug_img_count += 1

        return sample, np.array(target, dtype=np.float)#, count

    def __len__(self) -> int:
        return len(self.samples)

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
