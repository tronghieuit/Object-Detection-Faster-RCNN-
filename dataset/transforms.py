import random
import torch
from torchvision.transforms import functional as F
import albumentations as A
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Transform(object):

    def __call__(self, image, target):
        bbox = target["boxes"]
        label = target['labels']
        bbox = np.array(bbox)
        image = np.array(image)
        augmentation = A.Compose([
            A.HorizontalFlip(0.1),
            A.VerticalFlip(0.1),
            A.Blur(p=0.1),
            # A.RandomBrightnessContrast(p=0.1),
            # A.HueSaturationValue(p=0.1),
            # A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.1),
            # A.GaussNoise(var_limit=1.0 / 255.0, p=0.1)
        ],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
        )
        transformed = augmentation(image=image, bboxes=bbox, category_ids=label)
        bbox = list(transformed['bboxes'])
        image = transformed['image']
        label = transformed['category_ids']
        bbox = np.array([np.array(b) for b in bbox])
        bbox[:, 2] += bbox[:, 0]
        bbox[:, 3] += bbox[:, 1]
        target["boxes"] = torch.tensor(np.array(bbox))
        target['labels'] = torch.tensor(np.array(label))
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
