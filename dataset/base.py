import os
import random

import cv2
import numpy as np
from PIL import Image

from typing import Optional, Callable, Any, Tuple, Dict, List
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, ):
        self.root = None
        self.transform = None
        self.target_transform = None

        # default extensions support
        self.IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def is_valid_image(self, filename: str) -> bool:
        return self.has_file_allowed_extension(filename, self.IMG_EXTENSIONS)

    @staticmethod
    def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
        return filename.lower().endswith(extensions)

    # image loader e.g. PIL or OpenCV or IPP(inter)
    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    @staticmethod
    def opencv_loader(path: str) -> np.ndarray:
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def accimage_loader(self, path: str) -> Any:
        import accimage
        try:
            return accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return self.pil_loader(path)

    def default_loader(self, path: str, type: str = 'pil') -> Any:
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return self.accimage_loader(path)
        else:
            if type == 'opencv':
                return self.opencv_loader(path)
            else:
                return self.pil_loader(path)
