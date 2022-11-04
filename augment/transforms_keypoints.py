import numpy as np
import torch
import cv2
import imgaug as ia
from PIL import Image


class AugKeypoints(torch.nn.Module):

    def __init__(self, p, seq_det, convert_float_coord):
        super().__init__()
        self.p = p
        self.seq_det = seq_det.to_deterministic()
        self.convert_float_coord = convert_float_coord

    def forward(self, img, keypoints):
        """
        Args:
            img (PIL Image or Tensor): Image to be equalized.

        Returns:
            PIL Image or Tensor: Randomly equalized image.
        """
        self.seq_det.to_deterministic()
        if torch.rand(1).item() < self.p:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            new_keypoints = []
            cur_keypoints = []
            for cur_point in keypoints:
                cur_keypoints.append(ia.Keypoint(x=cur_point[0], y=cur_point[1]))
            images_aug = self.seq_det.augment_images([img])[0]
            key_points_on_Image = ia.KeypointsOnImage(cur_keypoints, shape=img.shape)
            keypoints_aug = self.seq_det.augment_keypoints([key_points_on_Image])[0]
            for i in range(len(key_points_on_Image.keypoints)):
                point_aug = keypoints_aug.keypoints[i]

                new_keypoints.append((np.array([point_aug.x, point_aug.y])).tolist())

            images_aug = cv2.cvtColor(images_aug, cv2.COLOR_BGR2RGB)
            images_aug = Image.fromarray(images_aug)
            return images_aug, new_keypoints
        return img, keypoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
