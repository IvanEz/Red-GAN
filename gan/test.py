"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from util import util

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        if opt.dataset_mode == 'brats':
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated[b])])
            visuals = visualizer.convert_visuals_to_numpy(visuals)
            image_numpy = visuals["synthesized_image"]
            util.save_image(image_numpy[:, :, 0], opt.results_dir + str(opt.condition_class) + '/train_t1ce_img_full'
                            + '/{}.png'.format(i * opt.batchSize + b),
                            create_dir=True)
            util.save_image(image_numpy[:, :, 1], opt.results_dir + str(opt.condition_class) + '/train_flair_img_full'
                            + '/{}.png'.format(i * opt.batchSize + b),
                            create_dir=True)
            util.save_image(image_numpy[:, :, 2], opt.results_dir + str(opt.condition_class) + '/train_t2_img_full'
                            + '/{}.png'.format(i * opt.batchSize + b),
                            create_dir=True)
            util.save_image(image_numpy[:, :, 3], opt.results_dir + str(opt.condition_class) + '/train_t1_img_full'
                            + '/{}.png'.format(i * opt.batchSize + b),
                            create_dir=True)
            print('processing t1ce, flair, t2, t1 modalities of index {}'.format(i * opt.batchSize + b))
        else:
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated[b])])
            visuals = visualizer.convert_visuals_to_numpy(visuals)
            image_numpy = visuals["synthesized_image"]
            util.save_image(image_numpy, opt.results_dir + str(opt.scanner_class)
                            + '/{}'.format(img_path[b].split('/')[-1]),
                            create_dir=True)

webpage.save()
