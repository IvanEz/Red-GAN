import nibabel as nib
import numpy as np
import os
from scipy import ndimage
from PIL import Image
import scipy

dataset_folder = './MICCAI_BraTS_2019_Data_Training/HGG/'
result_root = './MICCAI_BraTS_2019_Data_Training/HGG_res/'

results_dirs = {
    't1ce': './MICCAI_BraTS_2019_Data_Training/HGG_res/train_t1ce_img_full/',
    'flair': './MICCAI_BraTS_2019_Data_Training/HGG_res/train_flair_img_full/',
    't2': './MICCAI_BraTS_2019_Data_Training/HGG_res/train_t2_img_full/',
    't1': './MICCAI_BraTS_2019_Data_Training/HGG_res/train_t1_img_full/',
    'label': './MICCAI_BraTS_2019_Data_Training/HGG_res/train_label_full/'
}

scanner_classes = [
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

for item in results_dirs.values():
    os.mkdir(item)


def slice_scans():
    j = 0

    folders = os.listdir(dataset_folder)

    for idx, filename in enumerate(folders):

        for scanner_class in scanner_classes:
            if scanner_class in filename:
                scanner_class_file = scanner_class
                print(scanner_class_file)
                break

        image_folder = dataset_folder + filename
        labels = nib.load(image_folder + '/{}'.format(filename) + "_seg.nii.gz")
        labels = labels.get_data()

        # load the nii files
        t1c = nib.load(image_folder + '/{}'.format(filename) + "_t1ce.nii.gz")
        t1c = t1c.get_data()
        t1c = (t1c - np.amin(t1c)) / (np.amax(t1c) - np.amin(t1c))

        flair = nib.load(image_folder + '/{}'.format(filename) + "_flair.nii.gz")
        flair = flair.get_data()
        flair = (flair - np.amin(flair)) / (np.amax(flair) - np.amin(flair))

        t2 = nib.load(image_folder + '/{}'.format(filename) + "_t2.nii.gz")
        t2 = t2.get_data()
        t2 = (t2 - np.amin(t2)) / (np.amax(t2) - np.amin(t2))

        t1 = nib.load(image_folder + '/{}'.format(filename) + "_t1.nii.gz")
        t1 = t1.get_data()
        t1 = (t1 - np.amin(t1)) / (np.amax(t1) - np.amin(t1))

        # get only the slices with n tumor pixels in them
        count = np.sum(labels, axis=(0, 1))
        labels = labels[:, :, count > 1000]
        t1c = t1c[:, :, count > 1000]
        flair = flair[:, :, count > 1000]
        t2 = t2[:, :, count > 1000]
        t1 = t1[:, :, count > 1000]

        for slice_idx in range(t1c.shape[2]):
            mask, nr_objects = ndimage.label(t1c[:, :, slice_idx])

            c = np.logical_and(mask == 1, labels[:, :, slice_idx] == 0)
            labels[:, :, slice_idx][c == 1] = 3
            lbl_t1ce = Image.fromarray(labels[:, :, slice_idx])
            lbl_t1ce = lbl_t1ce.convert('L')

            # save the slices
            lbl_t1ce.save(results_dirs['label'] + "{}_".format(j) + scanner_class_file + ".png")
            scipy.misc.imsave(results_dirs['t1ce'] + "{}_".format(j) + scanner_class_file + ".png",
                              t1c[:, :, slice_idx])
            scipy.misc.imsave(results_dirs['flair'] + "{}_".format(j) + scanner_class_file + ".png",
                              flair[:, :, slice_idx])
            scipy.misc.imsave(results_dirs['t2'] + "{}_".format(j) + scanner_class_file + ".png",
                              t2[:, :, slice_idx])
            scipy.misc.imsave(results_dirs['t1'] + "{}_".format(j) + scanner_class_file + ".png",
                              t1[:, :, slice_idx])

            print(np.unique(labels[:, :, slice_idx]))

            j += 1


slice_scans()
