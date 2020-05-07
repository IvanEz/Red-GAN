from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import glob
import numpy as np
import json


class Dataset(BaseDataset):

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            scanner=None,
            synthesized=False,
            isic_meta_data='./data/isic/meta_data.json',
    ):
        self.classes = classes
        self.lesion_classes = [
            'melanoma',
            'seborrheic keratosis',
            'nevus'
        ]

        files = os.listdir(images_dir)
        format_img = ".png" if synthesized else ".jpg"

        ids = {key: [] for key in self.lesion_classes}
        with open(isic_meta_data, 'r') as f:
            meta_data = json.load(f)

        for meta in meta_data:
            diag = meta["meta"]["clinical"]["diagnosis"]
            if meta["name"] + format_img in files:
                ids[diag].append(meta["name"])

        if scanner is None:
            self.ids = [os.path.basename(x) for x in glob.glob(images_dir + r'/*.*')]
        else:
            self.ids = ids[scanner]

        self.images_fps = [os.path.join(images_dir, image_id.split('.')[0] + format_img)
                           for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0] + '_segmentation.png')
                          for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)
        mask = np.expand_dims(mask, axis=0).astype(np.float)
        mask = mask / 255.0

        return image, mask

    def __len__(self):
        return len(self.ids)
