[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Red-GAN
The implementation is in large adopted from SPADE-GAN (https://github.com/NVlabs/SPADE) 

### Installation
The following requirements should be installed for training the GAN code:
```torch>=1.0.0
torchvision
dominate>=2.3.1
dill
scikit-image
```
### Help
#### Train
The train.py script is used for training. Executing the help statement `python3 train.py -h` shows available training options:

````
usage: train.py [-h] [--gpu_ids GPU_IDS]                                                                                                                                                              
                [--checkpoints_dir CHECKPOINTS_DIR] [--model MODEL]                                 
                [--norm_G NORM_G] [--norm_D NORM_D] [--norm_E NORM_E] [--batchSize BATCHSIZE]        
                [--preprocess_mode {resize_and_crop,crop,scale_width,scale_width_and_crop,scale_shortside,scale_shortside_and_crop,fixed,none}]
                [--load_size LOAD_SIZE] [--crop_size CROP_SIZE]
                [--aspect_ratio ASPECT_RATIO] [--label_nc LABEL_NC]
                [--contain_dontcare_label] [--dataroot DATAROOT]
                [--dataset_mode DATASET_MODE] [--serial_batches] [--no_flip]
                [--nThreads NTHREADS] [--max_dataset_size MAX_DATASET_SIZE]
                [--load_from_opt_file] [--cache_filelist_write]
                [--cache_filelist_read] [--display_winsize DISPLAY_WINSIZE]
                [--netG NETG] [--ngf NGF] [--init_type INIT_TYPE]
                [--init_variance INIT_VARIANCE] [--z_dim Z_DIM] --segmentator
                SEGMENTATOR [--instance] [--nef NEF] [--use_vae]
                [--display_freq DISPLAY_FREQ] [--print_freq PRINT_FREQ]
                [--save_latest_freq SAVE_LATEST_FREQ]
                [--save_epoch_freq SAVE_EPOCH_FREQ] [--no_html] [--debug]
                [--tf_log] [--continue_train] [--which_epoch WHICH_EPOCH]
                [--niter NITER] [--niter_decay NITER_DECAY]
                [--optimizer OPTIMIZER] [--beta1 BETA1] [--beta2 BETA2]
                [--lr LR] [--D_steps_per_G D_STEPS_PER_G] [--ndf NDF]
                [--lambda_feat LAMBDA_FEAT] [--lambda_vgg LAMBDA_VGG]
                [--no_ganFeat_loss] [--vgg_loss] [--gan_mode GAN_MODE]
                [--netD NETD] [--no_TTUR] [--lambda_kld LAMBDA_KLD]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name of the experiment. It decides where to store
                        samples and models (default: label2coco)
  --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU (default:
                        0)
  --checkpoints_dir CHECKPOINTS_DIR  
                        models are saved here (default: ./checkpoints)
  --model MODEL         which model to use (default: pix2pix)
  --norm_G NORM_G       instance normalization or batch normalization
                        (default: spectralinstance)
  --norm_D NORM_D       instance normalization or batch normalization
                        (default: spectralinstance)
  --norm_E NORM_E       instance normalization or batch normalization
                        (default: spectralinstance)
  --phase PHASE         train, val, test, etc (default: train)
  --batchSize BATCHSIZE
                        input batch size (default: 1)
  --preprocess_mode {resize_and_crop,crop,scale_width,scale_width_and_crop,scale_shortside,scale_shortside_and_crop,fixed,none}
                        scaling and cropping of images at load time. (default:
                        none)
  --load_size LOAD_SIZE
                        Scale images to this size. The final image will be
                        cropped to --crop_size. (default: 1024)
  --crop_size CROP_SIZE
                        Crop to the width of crop_size (after initially
                        scaling the images to load_size.) (default: 512)
  --aspect_ratio ASPECT_RATIO
                        The ratio width/height. The final height of the load
                        image will be crop_size/aspect_ratio (default: 1.0)
  --label_nc LABEL_NC   # of input label classes without unknown class. If you
                        have unknown class as class label, specify
                        --contain_dopntcare_label. (default: 182)
  --contain_dontcare_label
                        if the label map contains dontcare label
                        (dontcare=255) (default: False)
  --dataroot DATAROOT
  --dataset_mode DATASET_MODE
  --serial_batches      if true, takes images in order to make batches,
                        otherwise takes them randomly (default: False)
  --no_flip             if specified, do not flip the images for data
                        argumentation (default: False)
  --nThreads NTHREADS   # threads for loading data (default: 0)
  --max_dataset_size MAX_DATASET_SIZE
                        Maximum number of samples allowed per dataset. If the
                        dataset directory contains more than max_dataset_size,
                        only a subset is loaded. (default:
                        9223372036854775807)
  --load_from_opt_file  load the options from checkpoints and use that as
                        default (default: False)
  --cache_filelist_write
                        saves the current filelist into a text file, so that
                        it loads faster (default: False)
  --cache_filelist_read
                        reads from the file list cache (default: False)
  --display_winsize DISPLAY_WINSIZE  
                        display window size (default: 400)
  --netG NETG           selects model to use for netG (pix2pixhd | spade)
                        (default: spade)
  --ngf NGF             # of gen filters in first conv layer (default: 64)
  --init_type INIT_TYPE
                        network initialization
                        [normal|xavier|kaiming|orthogonal] (default: xavier)
  --init_variance INIT_VARIANCE
                        variance of the initialization distribution (default:
                        0.02)
  --z_dim Z_DIM         dimension of the latent z vector (default: 256)
  --segmentator SEGMENTATOR
                        path to the segmentator network (default: None)
  --instance            if specified, add instance map as input (default:
                        False)
  --nef NEF             # of encoder filters in the first conv layer (default:
                        16)
  --use_vae             enable training with an image encoder. (default:
                        False)

  --display_freq DISPLAY_FREQ
                        frequency of showing training results on screen
                        (default: 100)
  --print_freq PRINT_FREQ
                        frequency of showing training results on console
                        (default: 100)
  --save_latest_freq SAVE_LATEST_FREQ
                        frequency of saving the latest results (default: 5000)
  --save_epoch_freq SAVE_EPOCH_FREQ
                        frequency of saving checkpoints at the end of epochs
                        (default: 10)
  --no_html             do not save intermediate training results to
                        [opt.checkpoints_dir]/[opt.name]/web/ (default: False)
  --debug               only do one epoch and displays at each iteration
                        (default: False)
  --tf_log              if specified, use tensorboard logging. Requires
                        tensorflow installed (default: False)
  --continue_train      continue training: load the latest model (default:
                        False)
  --which_epoch WHICH_EPOCH
                        which epoch to load? set to latest to use latest
                        cached model (default: latest)
  --niter NITER         # of iter at starting learning rate. This is NOT the
                        total #epochs. Totla #epochs is niter + niter_decay
                        (default: 50)
  --niter_decay NITER_DECAY
                        # of iter to linearly decay learning rate to zero
                        (default: 0)
  --optimizer OPTIMIZER
  --beta1 BETA1         momentum term of adam (default: 0.5)
  --beta2 BETA2         momentum term of adam (default: 0.999)
  --lr LR               initial learning rate for adam (default: 0.0002)
  --D_steps_per_G D_STEPS_PER_G
                        number of discriminator iterations per generator
                        iterations. (default: 1)
  --ndf NDF             # of discrim filters in first conv layer (default: 64)
  --lambda_feat LAMBDA_FEAT
                        weight for feature matching loss (default: 10.0)
  --lambda_vgg LAMBDA_VGG
                        weight for vgg loss (default: 10.0)
  --no_ganFeat_loss     if specified, do *not* use discriminator feature
                        matching loss (default: False)
  --vgg_loss            if specified, use VGG feature matching loss (default:
                        False)
  --gan_mode GAN_MODE   (ls|original|hinge) (default: hinge)
  --netD NETD           (n_layers|multiscale|image) (default: multiscale)
  --no_TTUR             Use TTUR training scheme (default: False)
  --lambda_kld LAMBDA_KLD
````
#### Test
The test.py script is used for evaluation. Executing the help statement `python3 test.py -h`, outputs:

````
usage: test.py [-h] [--name NAME] [--gpu_ids GPU_IDS]                                                                                                                                                              
               [--checkpoints_dir CHECKPOINTS_DIR] [--model MODEL]
               [--norm_G NORM_G] [--norm_D NORM_D] [--norm_E NORM_E]
               [--phase PHASE] [--batchSize BATCHSIZE]
               [--preprocess_mode {resize_and_crop,crop,scale_width,scale_width_and_crop,scale_shortside,scale_shortside_and_crop,fixed,none}]
               [--load_size LOAD_SIZE] [--crop_size CROP_SIZE]
               [--aspect_ratio ASPECT_RATIO] [--label_nc LABEL_NC]
               [--contain_dontcare_label] [--dataroot DATAROOT]
               [--dataset_mode DATASET_MODE] [--serial_batches] [--no_flip]
               [--nThreads NTHREADS] [--max_dataset_size MAX_DATASET_SIZE]
               [--load_from_opt_file] [--cache_filelist_write]
               [--cache_filelist_read] [--display_winsize DISPLAY_WINSIZE]
               [--netG NETG] [--ngf NGF] [--init_type INIT_TYPE]
               [--init_variance INIT_VARIANCE] [--z_dim Z_DIM] --segmentator
               SEGMENTATOR [--instance] [--nef NEF] [--use_vae]
               [--results_dir RESULTS_DIR] [--which_epoch WHICH_EPOCH]
               [--how_many HOW_MANY] [--condition_class CONDITION_CLASS]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name of the experiment. It decides where to store
                        samples and models (default: label2coco)
  --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU (default:
                        0)
  --checkpoints_dir CHECKPOINTS_DIR  
                        models are saved here (default: ./checkpoints)
  --model MODEL         which model to use (default: pix2pix)
  --norm_G NORM_G       instance normalization or batch normalization
                        (default: spectralinstance)
  --norm_D NORM_D       instance normalization or batch normalization
                        (default: spectralinstance)
  --norm_E NORM_E       instance normalization or batch normalization
                        (default: spectralinstance)
  --phase PHASE         train, val, test, etc (default: train)
  --batchSize BATCHSIZE
                        input batch size (default: 1)
  --preprocess_mode {resize_and_crop,crop,scale_width,scale_width_and_crop,scale_shortside,scale_shortside_and_crop,fixed,none}
                        scaling and cropping of images at load time. (default:
                        scale_width_and_crop)
  --load_size LOAD_SIZE
                        Scale images to this size. The final image will be
                        cropped to --crop_size. (default: 256)
  --crop_size CROP_SIZE
                        Crop to the width of crop_size (after initially
                        scaling the images to load_size.) (default: 256)
  --aspect_ratio ASPECT_RATIO
                        The ratio width/height. The final height of the load
                        image will be crop_size/aspect_ratio (default: 1.0)
  --label_nc LABEL_NC   # of input label classes without unknown class. If you
                        have unknown class as class label, specify
                        --contain_dopntcare_label. (default: 182)

````

### Additional Note:
If the parameter `--segmentator` is not set (by specifying the segmentator path), then the third player is not included in the architecture, which results in the vanilla [SPADE](https://github.com/NVlabs/SPADE/blob/master/README.md) architecture.
