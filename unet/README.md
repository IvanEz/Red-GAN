# U-Net based segmentation models
[![Build Status](https://travis-ci.com/qubvel/segmentation_models.pytorch.svg?branch=master)](https://travis-ci.com/qubvel/segmentation_models.pytorch) [![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg)](https://shields.io/)

The models are adopted from https://github.com/qubvel/segmentation_models.pytorch.

### Help

The training/testing can be carried out using the file run.py. Executing the statement `python3 run.py -h`, results in the following output:

```
usage: run.py [-h] --model_name MODEL_NAME [--fold FOLD]                                                                                                                                                           
              [--train_mode {real_only,synthetic_only,mixed}] [--gpu GPU]
              [--real_ratio REAL_RATIO] [--synthetic_ratio SYNTHETIC_RATIO]
              [--test_ratio TEST_RATIO] [--validation_ratio VALIDATION_RATIO]
              [--epochs EPOCHS] [--epochs_elapsed EPOCHS_ELAPSED] --root_dir
              ROOT_DIR --real_dir REAL_DIR --synthetic_dir SYNTHETIC_DIR
              --test_dir TEST_DIR [--t1ce T1CE] [--t2 T2] [--t1 T1]
              [--flair FLAIR] [--label LABEL] [--images IMAGES]
              [--masks MASKS] --mode {train,fine_tune,test,continue_train}
              [--encoder {resnet34}] [--activation {softmax}]
              [--test_class {0,1,2,3,4,5,6,7,8,9}] [--dataset {brats,isic}]
              [--device {cuda,cpu}] [--isic_meta_data ISIC_META_DATA]


optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        the model name for loading/saving
  --fold FOLD           the current fold of the dataset
  --train_mode {real_only,synthetic_only,mixed}
  --gpu GPU             the id of the GPU to be used
  --real_ratio REAL_RATIO
                        proportion of the real data to use
  --synthetic_ratio SYNTHETIC_RATIO
                        proportion of the synthetic data to use
  --test_ratio TEST_RATIO
                        proportion of the test data to use
  --validation_ratio VALIDATION_RATIO
                        proportion of train set for validation
  --epochs EPOCHS       number of epochs to train for
  --epochs_elapsed EPOCHS_ELAPSED
                        if continue train, then epochs already elapsed
  --root_dir ROOT_DIR   root dir where models, logs and plot are saved
  --real_dir REAL_DIR   root dir for the real data directory
  --synthetic_dir SYNTHETIC_DIR
                        root dir for the synthetic data directory
  --test_dir TEST_DIR   root dir for the synthetic data directory
  --t1ce T1CE           name for t1ce data directories
  --t2 T2               name for t1ce data directories
  --t1 T1               name for t1ce data directories
  --flair FLAIR         name for t1ce data directories
  --label LABEL         name for labels data directories
  --images IMAGES       name for image data directories of isic
  --masks MASKS         name for mask data directories of isic
  --mode {train,fine_tune,test,continue_train}
  --encoder {resnet34}
  --activation {softmax}
  --test_class {0,1,2,3,4,5,6,7,8,9}
                        test only a specific class
  --dataset {brats,isic}
  --device {cuda,cpu}
  --isic_meta_data ISIC_META_DATA
                        path of meta data file for isic

```

### Arguments

The train_mode argument can have three different values:

* real_only: Training with the real data only
* synthetic_only: Training using the synthetic data only
* mixed: Training using a mixture of real and synthetic data

The dataset paths are generated dynamically using according to the following template:
* real: `[real_dir]/[t1 or t2 or t1ce or flair or images or label]`
* synthetic: `[synthetic_dir]/[t1 or t2 or t1ce or flair or images or label]`
* test: `test_dir]/[t1 or t2 or t1ce or flair or images or label]`

### Additional Details
For BRATS dataset, the class information is read from the name of the file e.g. a t1ce image belonging to the TCIA05 class will be found at the path:
`$HOME/Red-GAN/unet/data/fold_3/train_flair_img_full/3081_TCIA05.png`

For ISIC dataset, the class information is saved in the file meta_data.json