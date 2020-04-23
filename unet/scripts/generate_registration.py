import nibabel as nib
import numpy as np
import os
from PIL import Image
import cv2

# In[7]:
np.set_printoptions(threshold=np.inf)

dataset_folder = './registration_data/'
output_folder = './registration_output/'

data = './train_label_full/'

registration_combinations = 19

# os.mkdir(DIR_img_t1ce)
# os.mkdir(DIR_img_t1)
# os.mkdir(DIR_img_t2)
# os.mkdir(DIR_label)
# os.mkdir(DIR_label_transformed)


def myfunc():
    j = 0
    imgs = os.listdir(data)
    for idx, filename in enumerate(os.listdir(dataset_folder)):

        image_folder = dataset_folder + filename
        labels = nib.load(image_folder)
        labels = labels.get_data()

        count = np.sum(labels, axis=(0, 1))
        labels = labels[:, :, count > 1000]

        print(filename)

        for slice_idx in range(labels.shape[2]):
            for i in range(registration_combinations):
                tumor_region = labels[:, :, slice_idx]
                img1 = cv2.imread(data + imgs[np.random.randint(low=0, high=len(imgs))], 0)
                while len(np.unique(img1)) == 1:
                    img1 = cv2.imread(data + imgs[np.random.randint(low=0, high=len(imgs))], 0)
                tumor_region = np.where(tumor_region != 0, tumor_region, img1)
                tumor_region = np.where(img1 == 0, img1, tumor_region)
                lbl_t1ce = Image.fromarray(tumor_region)
                lbl_t1ce = lbl_t1ce.convert("L")
                lbl_t1ce.save(output_folder + "{}.png".format(j))

                print(np.unique(tumor_region))

                j += 1


myfunc()
