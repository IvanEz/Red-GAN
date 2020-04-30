"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class BratsDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=5)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, required=True, help='directory that contains label images')
        parser.add_argument('--image_dir_t1ce', type=str, required=True, help='directory that contains t1ce modality')
        parser.add_argument('--image_dir_flair', type=str, required=True, help='directory that contains flair modality')
        parser.add_argument('--image_dir_t2', type=str, required=True, help='directory that contains t2 modality')
        parser.add_argument('--image_dir_t1', type=str, required=True, help='directory that contains t1 modality')
        parser.add_argument('--scanner_nc', type=int, default=10, help='the number of scanner classes')

        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_paths = dict()

        image_dir_t1ce = opt.image_dir_t1ce
        image_paths['t1ce'] = make_dataset(image_dir_t1ce, recursive=False, read_cache=True)

        image_dir_flair = opt.image_dir_flair
        image_paths['flair'] = make_dataset(image_dir_flair, recursive=False, read_cache=True)

        image_dir_t2 = opt.image_dir_t2
        image_paths['t2'] = make_dataset(image_dir_t2, recursive=False, read_cache=True)

        image_dir_t1 = opt.image_dir_t1
        image_paths['t1'] = make_dataset(image_dir_t1, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        return label_paths, image_paths, instance_paths
