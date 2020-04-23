import cv2
import os

paths = ['/home/qasima/segmentation_models.pytorch/isic2018/data/train/images',
         '/home/qasima/segmentation_models.pytorch/isic2018/data/train/segmentation_masks',
         '/home/qasima/segmentation_models.pytorch/isic2018/data/test/images',
         '/home/qasima/segmentation_models.pytorch/isic2018/data/test/segmentation_masks']

d_paths = ['/home/qasima/segmentation_models.pytorch/isic2018/data_resized/train/images',
           '/home/qasima/segmentation_models.pytorch/isic2018/data_resized/train/segmentation_masks',
           '/home/qasima/segmentation_models.pytorch/isic2018/data_resized/test/images',
           '/home/qasima/segmentation_models.pytorch/isic2018/data_resized/test/segmentation_masks']

size = (256, 256)

for path in paths:
    files = os.listdir(path)
    print(path)
    for file in files:
        print(file)
        img = cv2.imread(os.path.join(path, file))
        img = cv2.resize(src=img, dsize=size)
        cv2.imwrite(os.path.join(path, file), img)
