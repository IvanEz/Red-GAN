import nibabel as nib
import numpy as np
import os
from scipy import ndimage
from PIL import Image
import scipy

# In[7]:
np.set_printoptions(threshold=np.inf)

dataset_seg = './adni_seg/'
dataset_img = './adni_img/'
result_root = './train_label_full_coregistration/'

DIR_label = './train_label_full_coregistration/'

os.mkdir(DIR_label)


def myfunc():
    j = 0
    for filename in os.listdir(dataset_seg):
        count = 0

        img_name = filename.replace('_integer.nii', '')
        img_name = img_name + '_integer_img.nii'

        labels = nib.load(dataset_seg + filename)
        labels = labels.get_data()

        img = nib.load(dataset_img + img_name)
        img = img.get_data()

        for slice_idx in range(img.shape[2]):
            mask, nr_objects = ndimage.label(img[:, :, slice_idx])

            c = np.logical_and(mask == 1, labels[:, :, slice_idx] == 0)
            labels[:, :, slice_idx][c == 1] = 3
            lbl_t1ce = Image.fromarray(labels[:, :, slice_idx])
            lbl_t1ce = lbl_t1ce.convert('L')
            lbl_t1ce.save(DIR_label + "{}.png".format(j))

            j += 1
            count += 1

        print(filename, count)


myfunc()
