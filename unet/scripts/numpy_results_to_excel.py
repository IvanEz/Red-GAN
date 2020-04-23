import pandas as pd
import os
import numpy as np

root_dir = '/home/qasima/segmentation_models.pytorch/isic2018/'
results_file = os.path.join(root_dir, 'all_class_results.npy')
dest_file = os.path.join(root_dir, 'all_class_results.xlsx')

data = np.load(results_file)
data = pd.DataFrame(data)

lesion_classes = [
    'melanoma',
    'seborrheic keratosis',
    'nevus'
]

data.columns = lesion_classes
data.to_excel(dest_file, index=False)