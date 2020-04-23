from PIL import Image
import numpy as np
from scipy.misc import imsave


DIR_label = "./dataset/label/"
DIR_seg = "./dataset/seg/"
DIR_fused = "./dataset/fused/"

np.set_printoptions(threshold=np.inf)

for j in range(9270):
    img_lbl = Image.open(DIR_label + "{}.png".format(j))
    img_seg = Image.open(DIR_seg + "{}.png".format(j))
    img_lbl = np.array(img_lbl)
    img_seg = np.array(img_seg)
    img_seg[img_seg != 0] += 4
    img_fused = np.where(img_lbl == 3, img_seg, img_lbl)
    img_fused = Image.fromarray(img_fused.astype(np.uint8))
    img_fused.save(DIR_fused + "{}.png".format(j))
