import os
from shutil import copyfile

COREGISTERED = './adni_final_brats_to_brats_nearneigb_inter/'
DATA = './MICCAI_BraTS_2019_Data_Training/HGG/'

for idx, file in enumerate(os.listdir(COREGISTERED)):
    dir = file.split('__')[0]
    copyfile(COREGISTERED + file, DATA + dir + '/' + file)
    print(COREGISTERED + file + 'to' + DATA + dir + '/' + file)
