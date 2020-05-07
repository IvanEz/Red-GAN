from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import numpy as np
import glob


class Dataset(BaseDataset):
    CLASSES = ['bg', 't_2', 't_1', 'b', 't_3']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            scanner=None,
            synthesized=None,
            isic_meta_data=None,
    ):
        self.scanner_classes = [
            '2013',
            'CBICA',
            'TCIA01',
            'TCIA02',
            'TCIA03',
            'TCIA04',
            'TCIA05',
            'TCIA06',
            'TCIA08',
            'TMC'
        ]

        if scanner is not None:
            self.ids = [os.path.basename(x) for x in glob.glob(images_dir['t1ce'] + r'/*' +
                                                               self.scanner_classes[scanner] + '.png')]
        else:
            self.ids = [os.path.basename(x) for x in glob.glob(images_dir['t1ce'] + r'/*.png')]

        self.images_fps_t1ce = [os.path.join(images_dir['t1ce'], image_id) for image_id in self.ids]
        self.images_fps_flair = [os.path.join(images_dir['flair'], image_id) for image_id in self.ids]
        self.images_fps_t2 = [os.path.join(images_dir['t2'], image_id) for image_id in self.ids]
        self.images_fps_t1 = [os.path.join(images_dir['t1'], image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data for all three modalities, i.e. t1ce, t1 and t2
        image_t1ce = cv2.imread(self.images_fps_t1ce[i], cv2.IMREAD_GRAYSCALE)
        image_flair = cv2.imread(self.images_fps_flair[i], cv2.IMREAD_GRAYSCALE)
        image_t2 = cv2.imread(self.images_fps_t2[i], cv2.IMREAD_GRAYSCALE)
        image_t1 = cv2.imread(self.images_fps_t1[i], cv2.IMREAD_GRAYSCALE)

        image = np.stack([image_t1ce, image_flair, image_t2, image_t1], axis=-1).astype('float32')
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

        if image.shape[0] == 256:
            mask = cv2.copyMakeBorder(mask, 8, 8, 8, 8, cv2.BORDER_CONSTANT, (0, 0, 0))

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask_stacked = np.stack(masks, axis=-1).astype('float32')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_stacked)
            image, mask_stacked = sample['image'], sample['mask']

        mask_stacked = np.swapaxes(mask_stacked, 1, 2)
        mask_stacked = np.swapaxes(mask_stacked, 0, 1)
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)

        return image, mask_stacked

    def __len__(self):
        return len(self.ids)
