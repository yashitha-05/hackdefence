import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_MAP = {
    100: 0,
    200: 1,
    300: 2,
    500: 3,
    550: 4,
    600: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9,
}

def remap_mask(mask):
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for k, v in CLASS_MAP.items():
        remapped[mask == k] = v
    return remapped

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            ToTensorV2()
        ])

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.images[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = remap_mask(mask)
            augmented = self.transform(image=image, mask=mask)
            return augmented["image"], augmented["mask"]
        else:
            augmented = self.transform(image=image)
            return augmented["image"], self.images[idx]