import sys
import os

sys.path.append('/home/lee/PycharmProjects/cv_framework/utils')
import cv2
import torch
from PIL import Image
from torchvision import transforms
from loguru import logger
from utils.util import get_images


def _preprocess(image, input_h, input_w):
    # preprocess
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    transforms_ = transforms.Compose([
        transforms.Resize((input_h, input_w)),
        transforms.ToTensor(),
        normalize,
    ])
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_color = Image.fromarray(img)
    inputs = transforms_(img_color)
    inputs = inputs.unsqueeze(0)
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    return inputs


def test(args):
    logger.info('This is embedding-classification test.')
    test_data_path = os.path.join(str(args.data_root), "test")
    logger.info(f'Test dataset path:{test_data_path}')
    images = get_images(test_data_path)
    print(images)
