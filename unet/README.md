# U-Net based segmentation models
[![Build Status](https://travis-ci.com/qubvel/segmentation_models.pytorch.svg?branch=master)](https://travis-ci.com/qubvel/segmentation_models.pytorch) [![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg)](https://shields.io/)

The models are adopted from https://github.com/qubvel/segmentation_models.pytorch.

### Installation

The following requirements should met:
```scipy
numpy
albumentations
torch
opencv-python
matplotlib
torchvision==0.2.2
pretrainedmodels==0.7.4
torchnet==0.0.4
```

### Help

The training/testing can be carried out using the file run.py. Executing the help statement `python3 run.py -h` prints in the console the available options:

```
    --model_name (type=str, required=True, help='the model name for loading/saving')
    --fold (type=int, default=1, help='the current fold of the dataset')
    --train_mode (type=str, choices=['real_only', 'synthetic_only', 'mixed'], default='mixed')
    --gpu (type=str, default='0', help='the id of the GPU to be used')
    --real_ratio (type=float, default=1.0, help='proportion of the real data to use')
    --synthetic_ratio (type=float, default=1.0, help='proportion of the synthetic data to use')
    --test_ratio (type=float, default=1.0, help='proportion of the test data to use')
    --validation_ratio (type=float, default=0.1, help='proportion of train set for validation')
    --epochs (type=int, default=50, help='number of epochs to train for')
    --epochs_elapsed (type=int, default=0, help="if continue train, then epochs already elapsed")
    --root_dir (type=str, required=True, help="root dir where models, logs and plot are saved")
    --real_dir (type=str, required=True, help="root dir for the real data directory")
    --synthetic_dir (type=str, required=True, help="root dir for the synthetic data directory")
    --test_dir (type=str, required=True, help="root dir for the test data directory")
    --t1ce (type=str, help="name for t1ce data directories", default="train_t1ce_img_full")
    --t2 (type=str, help="name for t2 data directories", default="train_t2_img_full")
    --t1 (type=str, help="name for t1 data directories", default="train_t1_img_full")
    --flair (type=str, help="name for flair data directories", default="train_flair_img_full")
    --label (type=str, help="name for labels data directories", default="train_label_full")
    --images (type=str, help="name for image data directories of isic", default="images")
    --masks (type=str, help="name for mask data directories of isic", default="segmentation_masks")
    --mode (type=str, required=True, choices=['train', 'fine_tune', 'test', 'continue_train'])
    --encoder (type=str, choices=['resnet34'], default='resnet34')
    --activation (type=str, choices=['softmax'], default='softmax')
    --test_class (type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], default=None, help='test only a specific class')
    --dataset (type=str, choices=['brats', 'isic'], default='brats')
    --device (type=str, choices=['cuda', 'cpu'], default='cuda')
    --isic_meta_data (type=str, default='./data/isic/meta_data.json', help='path of meta data file for isic')
```

### Train/test arguments

To train/test the model specify `--mode` argument (e.g. 'train' or 'test'):

The dataset paths from where the training data will be read follow the following template:

* real: `[--real_dir]/[--t1 or --t2 or --t1ce or --flair or --images or --label]`
* synthetic: `[--synthetic_dir]/[--t1 or --t2 or --t1ce or --flair or --images or --label]`
* test: `[--test_dir]/[--t1 or --t2 or --t1ce or --flair or --images or --label]`

The argument `--root_dir` specifies the directory where models, logs and plot are saved.

### Additional Details
For BRATS dataset, the class information is read from the name of the file. For example, a t1ce image belonging to the TCIA05 class will be found at the path:
`$HOME/Red-GAN/unet/data/fold_3/train_flair_img_full/3081_TCIA05.png`

For ISIC dataset, the class information is saved in the file meta_data.json
