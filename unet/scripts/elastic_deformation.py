import os
from dpipe.medim.augmentation import elastic_transform
import cv2
import numpy as np


for idx, file in enumerate(os.listdir('./train_label_full')):
    img = cv2.imread('./train_label_full/'+file, 0)
    img_elastic = elastic_transform(img, 7, order=0)
    img = np.where(((img == 1) | (img == 2) | (img == 4)), 3, img)
    img_elastic = np.where(img_elastic == 3, 0, img_elastic)
    img_elastic = np.where(((img_elastic == 1) | (img_elastic == 2) | (img_elastic == 4)), img_elastic, img)
    cv2.imwrite('./train_label_full_elastic/'+'{}.png'.format(idx + 38750), img_elastic)
