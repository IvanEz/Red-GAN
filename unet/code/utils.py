import albumentations as albu
import cv2


def get_training_augmentation_padding():
    test_transform = [
        albu.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, (0, 0, 0))
    ]
    return albu.Compose(test_transform)


def get_training_augmentation_simple():
    test_transform = [
        albu.Rotate(limit=15, interpolation=1, border_mode=4, value=None, always_apply=False, p=0.5),
        albu.VerticalFlip(always_apply=False, p=0.5),
        albu.HorizontalFlip(always_apply=False, p=0.5),
        albu.Transpose(always_apply=False, p=0.5),
        albu.CenterCrop(height=200, width=200, always_apply=False, p=0.5),
        albu.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, (0, 0, 0))
    ]
    return albu.Compose(test_transform)