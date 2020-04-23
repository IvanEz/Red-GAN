from PIL import Image
import numpy as np
from scipy.misc import imsave

ROOT = "/home/ahmad/idp/"

DIR_REAL = ROOT + "train_t1ce_img_full/"
DIR_SEG = ROOT + "train_label_full/"
DIR_REAL_REGION = ROOT + "train_t1ce_real_region_full/"
DIR_SEG_REGION = ROOT + "train_label_region_full/"
DIR_MASK_REGION = ROOT + "train_mask_full/"

for f in range(9270):
	img = Image.open(DIR_SEG + "{}.png".format(f))
	img_real = Image.open(DIR_REAL + "{}.png".format(f))
	img = np.array(img)
	img_real = np.array(img_real)

	img_tumor = np.copy(img)

	tumor_region = False


	for x in np.nditer(img_tumor, op_flags=['readwrite']):
		if not tumor_region and (x != 3 and x != 0):
			tumor_region = True
		if tumor_region and (x == 3 or x == 0):
			tumor_region = False
		if not tumor_region:
			x[...] = 0
		else:
			x[...] = 1

	'''
	rows = np.any(img_tumor, axis=1)
	cols = np.any(img_tumor, axis=0)
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]

	rows = rmax - rmin
	cols = cmax - cmin

	print(rmin, rmax, cmin, cmax)

	img_tumor_region = img[rmin : rmax, cmin : cmax]
	'''

	img_tumor_region = np.where(img_tumor==0, img_tumor, img)
	img_real_region = np.where(img_tumor==0, img_tumor, img_real)
	img_tumor = img_tumor * 255.0
	imsave(DIR_MASK_REGION + "{}.png".format(f), img_tumor)
	imsave(DIR_SEG_REGION + "{}.png".format(f), img_tumor_region)
	imsave(DIR_REAL_REGION + "{}.png".format(f), img_real_region)