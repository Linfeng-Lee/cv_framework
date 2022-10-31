import json,os,sys,random

import cv2
import PIL
import numpy as np
# from torchvision.datasets.vision import VisionDataset
from PIL import Image
from dataset.folder_base import VisionDatasetBase
from dataset.folder_base import IMG_EXTENSIONS
from utils.util import make_dir



class DatasetFolder(VisionDatasetBase):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

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
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None,
                 nums_of_class=None,remove_no_json_sample_flag=False,convert_float_flag=False,
                 save_aug_img_flag=True,return_path=False,asymmetry_id=True,**kwargs):
        super(DatasetFolder, self).__init__(root)
        self.transform = transform
        self.keypoint_transform=target_transform
        self.return_path=return_path
        self.asymmetry_id=asymmetry_id
        classes, class_to_idx = self._find_classes(self.root)
        self.nums_of_class=nums_of_class

        samples = self.make_dataset_base(self.root, class_to_idx, extensions, is_valid_file, nums_of_class)
        self.save_aug_img_flag=save_aug_img_flag
        self.save_aug_img_count=0

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))
        if loader is None:
            self.loader=self.default_loader
        else:
            self.loader = loader
        self.extensions = extensions

        self.aug_data_save_path = "temp/aug/"
        make_dir(self.aug_data_save_path)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.keypoints_dict,self.new_samples,self.good_samples=self.deal_polygon_json(samples,convert_float_flag=convert_float_flag)
        if remove_no_json_sample_flag:
            samples=self.new_samples+self.good_samples
        self.samples = samples
        self.targets = [s[1] for s in samples]


    def deal_keypoint_json(self,samples,convert_float_flag):
        """
        samples:是分类文件夹里的所有图片数据
        convert_float_flag:是否把关键点转换成小数
        """
        new_samples=[]
        keypoints_dict={}
        for img_path in samples:
            # 进行读取keypoints的json文件,去除imageData,并且变为小数坐标
            keypoints_json_path=img_path[0][:-3]+'json'
            if os.path.exists(keypoints_json_path):
                keypoints_json = self.keypoint_loader(keypoints_json_path)
                #首先要进行关键点按label进行排序，因为labelme存储的标签可能不是从label开始的。
                imageHeight=keypoints_json.get("imageHeight")
                imageWidth=keypoints_json.get("imageWidth")
                sort_res=sorted(keypoints_json.get('shapes'),key=lambda x:x.get('label'))
                keypoints_list=[]
                for keypoint in sort_res:
                    if convert_float_flag:
                        keypoints_list.extend(np.array(keypoint['points'][0]))
                    else:
                        keypoints_list.extend(keypoint['points'][0])
                #处理成中心点跟面积大小
                if keypoints_list:
                    areas=((keypoints_list[2]-keypoints_list[0])*(keypoints_list[3]-keypoints_list[1]))/1600
                    center_x=((keypoints_list[2]-keypoints_list[0])/2+keypoints_list[0])/imageWidth
                    center_y=((keypoints_list[3]-keypoints_list[1])/2+keypoints_list[1])/imageHeight
                    keypoints_list.clear()
                    keypoints_list.extend([center_x,center_y,areas])
                keypoints_dict.update({keypoints_json.get("imagePath"):keypoints_list})
                new_samples.append(img_path)
        return keypoints_dict,new_samples

    def deal_polygon_json(self,samples,convert_float_flag):
        """
        samples:是分类文件夹里的所有图片数据
        convert_float_flag:是否把关键点转换成小数
        """
        new_samples=[]
        good_samples=[]
        keypoints_dict={}
        for img_path in samples:
            if img_path[1]==0:
                good_samples.append(img_path)
            # 进行读取keypoints的json文件,去除imageData,并且变为小数坐标
            keypoints_json_path=img_path[0][:-3]+'json'
            if os.path.exists(keypoints_json_path):
                cur_img_name=img_path[0].split("\\")[-1]
                keypoints_json = self.keypoint_loader(keypoints_json_path)
                #首先要进行关键点按label进行排序，因为labelme存储的标签可能不是从label开始的。
                sort_res=sorted(keypoints_json.get('shapes'),key=lambda x:x.get('label'))
                keypoints_list=[]
                keypoints_cls=[]
                for keypoint in sort_res:
                    if keypoint.get("shape_type") == "polygon":
                        polygon_keypoints = keypoint["points"]
                    else:
                        point1 = keypoint["points"][0]
                        point2 = [keypoint["points"][1][0], keypoint["points"][0][1]]
                        point3 = keypoint["points"][1]
                        point4 = [keypoint["points"][0][0], keypoint["points"][1][1]]
                        polygon_keypoints = [point1, point2, point3, point4]
                    keypoints_list.append(polygon_keypoints)
                    if self.asymmetry_id:
                        cur_idx=1
                    else:
                        cur_idx=self.class_to_idx.get(keypoint["label"])
                    if cur_idx is None:
                        print("json标签文件里的label跟文件夹命名的label名称不一致,文件{}。".format(img_path[0]))
                        raise
                    else:
                        keypoints_cls.append(cur_idx)

                #这个问题导致后面生产landmark的时候出现空，从而应发报错。
                if keypoints_json.get("imagePath")!=cur_img_name:
                    print("标签文件：{}出现的标签里的imagepath跟图片名称不一致。无法解析label数据。".format(cur_img_name))
                    continue
                keypoints_dict.update({keypoints_json.get("imagePath"):[keypoints_list,keypoints_cls]})
                new_samples.append(img_path)
        return keypoints_dict,new_samples,good_samples

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
        with open(path,'r')as f:
            keypoints_json=json.load(f)
        return keypoints_json

    def landmark_to_mask_vec(self,mask,keypoints_list,class_id=1):
        mask=Image.fromarray(mask)
        draw=PIL.ImageDraw.Draw(mask)
        xy=[tuple(point)for point in keypoints_list]
        assert len(xy)!=1
        draw.polygon(xy=xy,outline=0,fill=class_id)
        mask=np.array(mask,dtype=np.uint8)
        return mask


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        cur_keypoint_info=self.keypoints_dict.get(path.split("\\")[-1])
        if cur_keypoint_info is not None:
            landmark,keypoint_cls=cur_keypoint_info
        else:
            landmark=cur_keypoint_info
        # keypoint_cls=self.keypoints_dict.get("classes")
        if landmark is not None:
            landmark_len = list(map(lambda x: len(x), landmark))
            if len(landmark_len) >= 1:
                new_landmark = []
                for i in range(len(landmark_len)):
                    new_landmark.extend(landmark[i])
                landmark = new_landmark.copy()

            #要转换成np,如果是list的话，会导致其得到的数据维度不一样。
            landmark=np.asarray(landmark,dtype=np.float32)

        if self.keypoint_transform is not None:
            if landmark is not None:
                sample,landmark= self.keypoint_transform(sample,landmark)
            else:
                sample,landmark=self.keypoint_transform(sample,[[0.0,0.0],[0.0,0.0]])
                landmark=None
            #启动训练服务的时候进行一次代码执行，查看其增强效果，从而可以保证增强是否可行
            if self.save_aug_img_count<100 and landmark:
                aug_path = self.aug_data_save_path + os.path.basename(path)[:-4] + "_aug.png"
                img = np.array(sample)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                start_len=0
                for idx,landmark_len_item in enumerate(landmark_len):
                    cur_landmark=landmark[start_len:(start_len+landmark_len_item)]
                    start_len+=landmark_len_item
                    cv2.drawContours(img, [np.array(cur_landmark, dtype=np.int32)], -1, (255, 0, 0), 1)
                cv2.imwrite(aug_path, img)
                self.save_aug_img_count+=1

            #显示增强后的效果
            if False:
                img = np.array(sample)
                cv2.drawContours(img, [np.array(landmark, dtype=np.int32)], -1, (255, 0, 0), 3)
                cv2.imwrite('temp/aug.jpg', img)
            if landmark is not None:
                mask = np.zeros((sample.size[1],sample.size[0]), dtype=np.uint8)
                start_len=0
                for idx,landmark_len_item in enumerate(landmark_len):
                    cur_landmark=landmark[start_len:(start_len+landmark_len_item)]
                    start_len+=landmark_len_item
                    cur_keypoint_class=keypoint_cls[idx]
                    mask=self.landmark_to_mask_vec(mask,cur_landmark,cur_keypoint_class)
                landmark=mask.copy()
            else:
                landmark=np.zeros((sample.size[1],sample.size[0]))
            landmark=np.array(landmark,dtype=np.float32)
            landmark=landmark[np.newaxis,:,:]
        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_path:
            return sample, target,landmark,path

        return sample, target,landmark

    def __len__(self):
        return len(self.samples)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None, nums_of_class=None,remove_no_json_sample_flag=False,convert_float_flag=False,return_path=False,
                 **kwargs):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, nums_of_class=nums_of_class,
                                          remove_no_json_sample_flag=remove_no_json_sample_flag,
                                          convert_float_flag=convert_float_flag,return_path=return_path,
                                          **kwargs)
        self.imgs = self.samples
