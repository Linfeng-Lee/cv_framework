import sys, os, os.path, json, random, pathlib
import numpy as np
import datetime
import cv2
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys

sys.path.append('/home/lee/PycharmProjects/cv_framework/augment')
from augment import aug_transforms
from augment import aug_config
from dataset.folder_base import VisionDatasetBase
from dataset.folder_base import IMG_EXTENSIONS
from dataset.base import BaseDataset
from dataset.classification import BaseDatasetClassify


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
    with open(filename, 'w', encoding="utf-8") as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)


class BaseMultilabelClassification(BaseDatasetClassify):
    def __init__(self,
                 root: str,
                 loader: Callable,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 extensions: Optional[Tuple[str, ...]] = None
                 ):
        super(BaseMultilabelClassification, self).__init__(root=root,
                                                           loader=loader,
                                                           transform=transform,
                                                           target_transform=target_transform,
                                                           extensions=extensions
                                                           )

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @staticmethod
    def get_label_head_digit(label: str) -> str:
        digit_str = str(label.split("_")[0])

        return digit_str

    # read the json file
    @staticmethod
    def keypoint_loader(path: str):
        with open(path, 'r', encoding="utf-8") as f:
            keypoints_json = json.load(f)
        return keypoints_json


# impl __getitem__() and __len__()
class ImplDatasetMultilabelClassification(BaseMultilabelClassification):
    def __init__(self,
                 root: str,
                 loader: Callable,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 extensions: Optional[Tuple[str, ...]] = None
                 ):
        super(ImplDatasetMultilabelClassification, self).__init__(root=root,
                                                                  loader=loader,
                                                                  transform=transform,
                                                                  target_transform=target_transform,
                                                                  extensions=extensions
                                                                  )

        self.is_valid_file = is_valid_file
        if self.is_valid_file:
            self.reload_data_path = os.path.join(root, "test.json")
        else:
            self.reload_data_path = os.path.join(root, "train.json")

        self.samples, self.classes, self.class_to_count, self.class_to_idx = self.load_data(self.root)
        self.save_aug_img_count = 0
        self.aug_data_save_path = "temp/aug/"

    def __getitem__(self, index: int) -> Any:
        cur_info = self.samples[index]
        sample = self.loader(cur_info["img_path"])

        target = len(self.classes) * [0]
        for label in cur_info["label"]:
            target[int(label)] = 1

        if self.transform is not None:
            sample = self.transform(sample)

        # if self.save_aug_img_count < 100 and not self.is_valid_file:
        #     aug_path = self.aug_data_save_path + os.path.basename(cur_info["img_path"])[:-4] + "_aug.png"
        #     norm = self.transform.transforms_.transforms[1]
        #     normalize_transpose = aug_transforms.NormalizeTranspose(mean=norm.mean, std=norm.std)
        #     img_aug = normalize_transpose(sample)
        #     img_aug = np.array(img_aug.cpu().detach().numpy(), dtype=np.uint8)
        #     img_aug = cv2.cvtColor(img_aug.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
        #     # cv2.imwrite(aug_path, img_aug)
        #     self.save_aug_img_count += 1

        return sample, np.array(target, dtype=np.float)  # , count

    def __len__(self) -> int:
        return len(self.samples)

    def get_images(self, path: str, **kwargs) -> list:
        images = []
        path = os.path.expanduser(path)

        extensions = kwargs.get('extensions')
        if extensions is not None:
            self.IMG_EXTENSIONS = extensions

        for root, dirs, files in os.walk(path):
            for file in files:
                if self.is_valid_image(file):
                    file = os.path.join(root, file)
                    images.append(file)
        random.shuffle(images)
        return images

    def load_data(self, data_path: str):
        """
        根据ignore里的text进行去除相应的字符信息。
        """
        data_list = self.get_images(data_path)
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

                class_to_idx.setdefault(label_digit, label)
                class_to_count.setdefault(label_digit, 0)

                if val:
                    class_to_count[label_digit] += 1
                    labelsdic[str(label_digit)] = label

            if not len(labelsdic):
                print('there is no label in {}'.format(img_path))
                continue

            item = {
                'img_path': img_path,
                'img_name': pathlib.Path(img_path).stem,
                'label': labelsdic
            }
            t_data_list.append(item)
            classes.extend(list(labelsdic.keys()))
            classes = list(set(classes))

        class_to_idx = dict(sorted(class_to_idx.items(), key=lambda x: x[0]))
        class_to_idx = dict(zip(class_to_idx.values(), class_to_idx.keys()))
        classes = list(class_to_idx.keys())

        return t_data_list, classes, class_to_count, class_to_idx


class DatasetMultilabelClassification(ImplDatasetMultilabelClassification):
    def __init__(self,
                 root: str,
                 loader: Callable,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super(DatasetMultilabelClassification, self).__init__(root=root,
                                                              loader=loader,
                                                              transform=transform,
                                                              target_transform=target_transform,
                                                              )
