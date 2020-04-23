import cv2
import numpy as np
import os
import nibabel as nib

result_root = '/home/ahmad/idp/deep_pipe/example/MICCAI_BraTS_2019_Data_Training/HGG_res_test/'
DIR_img_t1ce = result_root + '/train_t1ce_img_full/'
DIR_img_flair = result_root + '/train_flair_img_full/'
DIR_img_t2 = result_root + '/train_t2_img_full/'
DIR_img_t1 = result_root + '/train_t1_img_full/'
DIR_label = result_root + '/train_label_full/'
DIR_img_t1ce_3D = result_root + '/train_t1ce_img_full_3D/'
DIR_img_flair_3D = result_root + '/train_flair_img_full_3D/'
DIR_img_t2_3D = result_root + '/train_t2_img_full_3D/'
DIR_img_t1_3D = result_root + '/train_t1_img_full_3D/'
DIR_label_3D = result_root + '/train_label_full_3D/'
MAX_SLICES = 155
NUM_SCANS = 10

os.mkdir(DIR_img_t1ce_3D)
os.mkdir(DIR_img_flair_3D)
os.mkdir(DIR_img_t2_3D)
os.mkdir(DIR_img_t1_3D)
os.mkdir(DIR_label_3D)

for idx in range(NUM_SCANS):
    img_3D_t1ce = np.zeros((240, 240, MAX_SLICES), dtype=np.int8)
    img_3D_flair = np.zeros((240, 240, MAX_SLICES), dtype=np.int8)
    img_3D_t2 = np.zeros((240, 240, MAX_SLICES), dtype=np.int8)
    img_3D_t1 = np.zeros((240, 240, MAX_SLICES), dtype=np.int8)
    img_3D_label = np.zeros((240, 240, MAX_SLICES), dtype=np.int8)

    for x in range(MAX_SLICES):
        img_t1ce = cv2.imread(DIR_img_t1ce + '{}.png'.format(x + idx*MAX_SLICES), cv2.IMREAD_GRAYSCALE)
        img_flair = cv2.imread(DIR_img_flair + '{}.png'.format(x + idx*MAX_SLICES), cv2.IMREAD_GRAYSCALE)
        img_t2 = cv2.imread(DIR_img_t2 + '{}.png'.format(x + idx*MAX_SLICES), cv2.IMREAD_GRAYSCALE)
        img_t1 = cv2.imread(DIR_img_t1 + '{}.png'.format(x + idx*MAX_SLICES), cv2.IMREAD_GRAYSCALE)
        img_label = cv2.imread(DIR_label + '{}.png'.format(x + idx*MAX_SLICES), cv2.IMREAD_GRAYSCALE)

        img_3D_t1ce[:, :, x] = img_t1ce
        img_3D_flair[:, :, x] = img_flair
        img_3D_t2[:, :, x] = img_t2
        img_3D_t1[:, :, x] = img_t1
        img_3D_label[:, :, x] = img_label

    img_3D_t1ce_nib = nib.Nifti1Image(img_3D_t1ce, affine=None)
    nib.save(img_3D_t1ce_nib, DIR_img_t1ce_3D + '{}'.format(idx))
    img_3D_flair_nib = nib.Nifti1Image(img_3D_flair, affine=None)
    nib.save(img_3D_flair_nib, DIR_img_flair_3D + '{}'.format(idx))
    img_3D_t2_nib = nib.Nifti1Image(img_3D_t2, affine=None)
    nib.save(img_3D_t2_nib, DIR_img_t2_3D + '{}'.format(idx))
    img_3D_t1_nib = nib.Nifti1Image(img_3D_t1, affine=None)
    nib.save(img_3D_t1_nib, DIR_img_t1_3D + '{}'.format(idx))
    img_3D_label_nib = nib.Nifti1Image(img_3D_label, affine=None)
    nib.save(img_3D_label_nib, DIR_label_3D + '{}'.format(idx))
