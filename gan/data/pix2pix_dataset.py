"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import numpy as np
import os
import torch


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, instance_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        # modification: check if the dataset mode is brats
        if opt.dataset_mode == 'brats':
            util.natural_sort(image_paths['t1ce'])
            util.natural_sort(image_paths['flair'])
            util.natural_sort(image_paths['t2'])
            util.natural_sort(image_paths['t1'])
        else:
            util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        if opt.dataset_mode == 'brats':
            image_paths['t1ce'] = image_paths['t1ce'][:opt.max_dataset_size]
            image_paths['flair'] = image_paths['flair'][:opt.max_dataset_size]
            image_paths['t2'] = image_paths['t2'][:opt.max_dataset_size]
            image_paths['t1'] = image_paths['t1'][:opt.max_dataset_size]
        else:
            image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are " \
                    "quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see " \
                    "what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        # check if the the number of scanner classes given is the same as initialized in base_dataset file
        if opt.dataset_mode == "brats":
            assert len(self.scanner_classes) == opt.scanner_nc, "The number of scanner classes mismatch"

        self.label_paths = label_paths
        if opt.dataset_mode == 'brats':
            self.image_paths = dict()
            self.image_paths['t1ce'] = image_paths['t1ce']
            self.image_paths['flair'] = image_paths['flair']
            self.image_paths['t2'] = image_paths['t2']
            self.image_paths['t1'] = image_paths['t1']
        else:
            self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        label = label.convert('L')
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        # modification: scale by 255.0 because the tensor is normalized
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        if self.opt.dataset_mode == 'brats':
            image_path = dict()

            image_path['t1ce'] = self.image_paths['t1ce'][index]
            image_path['flair'] = self.image_paths['flair'][index]
            image_path['t2'] = self.image_paths['t2'][index]
            image_path['t1'] = self.image_paths['t1'][index]

            for idx, scanner_class in enumerate(self.scanner_classes):
                if self.opt.isTrain:
                    if scanner_class in image_path['flair']:
                        scanner_class_idx = idx
                        break
                else:
                    scanner_class_idx = self.opt.scanner_class

            image_t1ce = Image.open(image_path['t1ce'])
            image_flair = Image.open(image_path['flair'])
            image_t2 = Image.open(image_path['t2'])
            image_t1 = Image.open(image_path['t1'])
            transform_image = get_transform(self.opt, params)
            image_tensor_t1ce = transform_image(image_t1ce)
            image_tensor_flair = transform_image(image_flair)
            image_tensor_t2 = transform_image(image_t2)
            image_tensor_t1 = transform_image(image_t1)
            image_tensor = torch.cat((image_tensor_t1ce, image_tensor_flair, image_tensor_t2, image_tensor_t1), dim=0)


        else:
            # input image (real images)
            image_path = self.image_paths[index]
            # modification: removing this assert as the label filename doesn't have to the same as the image name
            # assert self.paths_match(label_path, image_path), \
            #    "The label_path %s and image_path %s don't match." % \
            #    (label_path, image_path)
            image = Image.open(image_path)
            # modification: convert the image to grayscale rather then RGB
            image = image.convert('L')
            transform_image = get_transform(self.opt, params)
            image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'scanner': scanner_class_idx,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
