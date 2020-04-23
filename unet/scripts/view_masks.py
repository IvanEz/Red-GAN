import os
import cv2
import numpy as np

data = "./masks/"
scale = True
recolor = True


for idx, path in enumerate(os.listdir(data)):
    img = cv2.imread(data + path, 0)
    '''
    if scale:
        cv2.imwrite(data + path, (img/np.max(img))*255)
    else:
        img = np.where(((img == 1) | (img == 2) | (img == 4)), 3, img)
        cv2.imwrite(data + path, img)
	'''
    if recolor:
        img_rgb = np.zeros((*img.shape, 3))
        img_rgb[:, :, 0] = np.where(img == 3, 107, img_rgb[:, :, 0])
        img_rgb[:, :, 0] = np.where(img == 1, 255, img_rgb[:, :, 0])
        img_rgb[:, :, 1] = np.where(img == 3, 107, img_rgb[:, :, 1])
        img_rgb[:, :, 1] = np.where(img == 2, 255, img_rgb[:, :, 1])
        img_rgb[:, :, 2] = np.where(img == 3, 107, img_rgb[:, :, 2])
        img_rgb[:, :, 2] = np.where(img == 4, 255, img_rgb[:, :, 2])
        cv2.imwrite(path, img_rgb)
