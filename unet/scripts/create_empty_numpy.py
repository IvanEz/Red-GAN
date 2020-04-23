import numpy as np
import os

root_dir = '/home/qasima/segmentation_models.pytorch/isic2018/'
file = os.path.join(root_dir, 'all_class_results.npy')

data = np.zeros(shape=(3, ))

np.save(file, data)