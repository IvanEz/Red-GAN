import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
import albumentations as albu
import pickle
import matplotlib.pyplot as plt
from brats_dataset import Dataset
from configs import configs
from scipy.stats import wilcoxon, ttest_ind
import segmentation_models_pytorch as smp
import sys

"""
Using a U-net architecture for segmentation of Tumor Modalities
"""


class UnetTumorSegmentator:
    def __init__(self, mode, model_name, pure_ratio, synthetic_ratio, augmented_ratio):

        # the fold number to execute the code for
        self.fold = 'fold_4'

        self.root_dir = '/home/qasima/segmentation_models.pytorch/'
        self.data_dir = '/home/qasima/segmentation_models.pytorch/data/' + self.fold + '/'

        # epochs
        self.epochs_num = 50
        self.total_epochs = 0 + self.epochs_num
        self.continue_train = False

        # what proportion of pure data to be used
        self.pure_ratio = pure_ratio

        # proportion of synthetic or augmented data to be used, depending on the mode, w.r.t the pure dataset size
        # can be set upto 2.0 i.e. 200%
        self.synthetic_ratio = synthetic_ratio
        self.augmented_ratio = augmented_ratio

        # proportion of test data to be used
        self.test_ratio = 1.0

        # test and validation ratios to use
        self.validation_ratio = 0.1

        # training
        # mode = pure, none_only
        self.mode = mode
        self.encoder = 'resnet34'
        self.encoder_weights = None
        self.device = 'cuda'
        self.activation = 'softmax'
        self.loss = 'cross_entropy'

        self.all_classes = ['bg', 't_2', 't_1', 'b', 't_3']
        # classes to be trained upon
        self.classes = ['t_2', 't_1', 't_3']

        # paths
        self.model_name = model_name
        self.log_dir = self.root_dir + 'logs/' + self.loss + '/' + self.fold + '/' + self.model_name
        self.model_root = self.root_dir + 'models/' + self.loss + '/' + self.fold + '/'
        self.model_dir = self.model_root + self.model_name
        self.result_dir = self.root_dir + 'results/' + self.loss + '/' + self.fold + '/' + self.model_name
        self.scanner_plots_paths = self.root_dir + "scanner_class_plots/" + self.loss + '/' + self.fold + '/'

        # dataset paths
        self.x_dir = dict()
        self.y_dir = None
        self.x_dir_syn = dict()
        self.y_dir_syn = None
        self.x_dir_test = dict()
        self.y_dir_test = None

        # loaded or created model
        self.model = None

        # full dataset and the pure dataset
        self.full_dataset = None
        self.full_dataset_pure = None

        # model setup
        self.model_loss = None
        self.metrics = None
        self.optimizer = None

        # scanners
        self.scanner_classes = [
            '2013',
            'CBICA',
            'TCIA01',
            'TCIA02',
            'TCIA03',
            'TCIA04',
            'TCIA05',
            'TCIA06',
            'TCIA08',
            'TMC',
        ]
        self.fine_tuning = False
        self.scanner_class = 0
        self.scanner_classes_size = []

        self.freeze_layers_encoder = None
        self.fine_tune_layers_encoder = None

        self.freeze_layers_decoder = None
        self.fine_tune_layers_decoder = None

    def create_folders(self):
        # create folders for results and logs to be saved
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.isdir(self.model_root):
            os.makedirs(self.model_root)
        if not os.path.isdir(self.scanner_plots_paths):
            os.makedirs(self.scanner_plots_paths)

    def set_dataset_paths(self):
        # pure dataset
        self.x_dir['t1ce'] = os.path.join(self.data_dir, 'train_t1ce_img_full')
        self.x_dir['flair'] = os.path.join(self.data_dir, 'train_flair_img_full')
        self.x_dir['t2'] = os.path.join(self.data_dir, 'train_t2_img_full')
        self.x_dir['t1'] = os.path.join(self.data_dir, 'train_t1_img_full')
        self.y_dir = os.path.join(self.data_dir, 'train_label_full')

        # set the synthetic dataset paths
        if 'none' in self.mode or 'none_only' in self.mode:
            # none mode or while training on none masks only
            self.x_dir_syn['t1ce'] = os.path.join(self.data_dir, '{}/train_t1ce_img_full'.
                                                  format(self.scanner_class))
            self.x_dir_syn['flair'] = os.path.join(self.data_dir, '{}/train_flair_img_full'.
                                                   format(self.scanner_class))
            self.x_dir_syn['t2'] = os.path.join(self.data_dir, '{}/train_t2_img_full'.
                                                format(self.scanner_class))
            self.x_dir_syn['t1'] = os.path.join(self.data_dir, '{}/train_t1_img_full'.
                                                format(self.scanner_class))
            self.y_dir_syn = os.path.join(self.data_dir, '{}/train_label_full'.
                                          format(self.scanner_class))

        elif self.mode == 'fine_tune_scanner' or self.mode == 'train_scanner':
            if self.scanner_class == "balanced":
                self.x_dir_syn['t1ce'] = list()
                self.x_dir_syn['flair'] = list()
                self.x_dir_syn['t2'] = list()
                self.x_dir_syn['t1'] = list()
                self.y_dir_syn = list()

                for i, cls in enumerate(self.scanner_classes):
                    self.x_dir_syn['t1ce'].append(os.path.join(self.data_dir, '{}/train_t1ce_img_full'.
                                                               format(i)))
                    self.x_dir_syn['flair'].append(os.path.join(self.data_dir, '{}/train_flair_img_full'.
                                                                format(i)))
                    self.x_dir_syn['t2'].append(os.path.join(self.data_dir, '{}/train_t2_img_full'.
                                                             format(i)))
                    self.x_dir_syn['t1'].append(os.path.join(self.data_dir, '{}/train_t1_img_full'.
                                                             format(i)))
                    self.y_dir_syn.append(os.path.join(self.data_dir, '{}/train_label_full'.
                                                       format(i)))
            else:
                # scanner class dataset
                self.x_dir_syn['t1ce'] = os.path.join(self.data_dir, '{}/train_t1ce_img_full'.
                                                      format(self.scanner_class))
                self.x_dir_syn['flair'] = os.path.join(self.data_dir, '{}/train_flair_img_full'.
                                                       format(self.scanner_class))
                self.x_dir_syn['t2'] = os.path.join(self.data_dir, '{}/train_t2_img_full'.
                                                    format(self.scanner_class))
                self.x_dir_syn['t1'] = os.path.join(self.data_dir, '{}/train_t1_img_full'.
                                                    format(self.scanner_class))
                self.y_dir_syn = os.path.join(self.data_dir, '{}/train_label_full'.
                                              format(self.scanner_class))

        # test dataset
        self.x_dir_test['t1ce'] = os.path.join(self.data_dir, 'train_t1ce_img_full_test')
        self.x_dir_test['flair'] = os.path.join(self.data_dir, 'train_flair_img_full_test')
        self.x_dir_test['t2'] = os.path.join(self.data_dir, 'train_t2_img_full_test')
        self.x_dir_test['t1'] = os.path.join(self.data_dir, 'train_t1_img_full_test')
        self.y_dir_test = os.path.join(self.data_dir, 'train_label_full_test')

    def scanner_class_sizes(self):
        for i, cls in enumerate(self.scanner_classes):
            scanner_dataset = Dataset(
                self.x_dir,
                self.y_dir,
                classes=self.classes,
                scanner=i,
                augmentation=self.get_training_augmentation_padding(),
            )
            self.scanner_classes_size.append(len(scanner_dataset))

    def create_dataset(self):
        # create the pure dataset
        self.full_dataset_pure = Dataset(
            self.x_dir,
            self.y_dir,
            classes=self.classes,
            augmentation=self.get_training_augmentation_padding(),
        )

        pure_size = int(len(self.full_dataset_pure) * self.pure_ratio)
        self.full_dataset_pure = torch.utils.data.Subset(self.full_dataset_pure, np.arange(pure_size))

        if self.mode == 'pure':
            # mode is pure then full dataset is the pure dataset
            self.full_dataset = self.full_dataset_pure

        elif "none" in self.mode:
            # mode is none_only, full dataset consists of synthetic images generated from segmentation masks without
            # any augmentations
            full_dataset_syn = Dataset(
                self.x_dir_syn,
                self.y_dir_syn,
                classes=self.classes,
                augmentation=self.get_training_augmentation_simple(),
            )

            # synthetic_size = int(len(self.full_dataset_pure) * self.synthetic_ratio)
            self.full_dataset = full_dataset_syn
            self.full_dataset_pure = self.full_dataset

        elif self.mode == "train_scanner":
            if self.scanner_class == "balanced":
                datasets = []
                max_scanner_class = max(self.scanner_classes_size)

                for i, cls in enumerate(self.scanner_classes):
                    x_dir_syn = {'t1ce': self.x_dir_syn['t1ce'][i],
                                 't2': self.x_dir_syn['t2'][i],
                                 't1': self.x_dir_syn['t1'][i],
                                 'flair': self.x_dir_syn['flair'][i]}
                    y_dir_syn = self.y_dir_syn[i]

                    full_dataset_scanner_syn = Dataset(
                        x_dir_syn,
                        y_dir_syn,
                        classes=self.classes,
                        augmentation=self.get_training_augmentation_padding(),
                    )

                    full_dataset_scanner_syn = torch.utils.data.Subset(full_dataset_scanner_syn,
                                                                       np.arange(max_scanner_class -
                                                                                 self.scanner_classes_size[i]))

                    datasets.append(full_dataset_scanner_syn)

                full_dataset_syn = torch.utils.data.ConcatDataset(datasets)
            else:
                full_dataset_syn = Dataset(
                    self.x_dir_syn,
                    self.y_dir_syn,
                    classes=self.classes,
                    augmentation=self.get_training_augmentation_padding(),
                )

            synthetic_size = int(len(full_dataset_syn) * self.synthetic_ratio)
            full_dataset_syn = torch.utils.data.Subset(full_dataset_syn, np.arange(synthetic_size))

            # 200%
            # full_dataset_syn = torch.utils.data.ConcatDataset((full_dataset_syn, full_dataset_syn))
            self.full_dataset = torch.utils.data.ConcatDataset((self.full_dataset_pure, full_dataset_syn))

        else:
            # for modes elastic, coregistration and none simply add the corresponding synthetic images to pure dataset
            full_dataset_syn = Dataset(
                self.x_dir_syn,
                self.y_dir_syn,
                classes=self.classes,
                augmentation=self.get_training_augmentation_padding(),
            )

            synthetic_size = int(len(full_dataset_syn) * self.synthetic_ratio)
            full_dataset_syn = torch.utils.data.Subset(full_dataset_syn, np.arange(synthetic_size))

            # 200%
            # full_dataset_syn = torch.utils.data.ConcatDataset((full_dataset_syn, full_dataset_syn))
            self.full_dataset = torch.utils.data.ConcatDataset((self.full_dataset_pure, full_dataset_syn))

        return self.full_dataset, self.full_dataset_pure

    def create_model(self):
        # create or load the model
        if self.continue_train:
            self.model = torch.load(self.model_dir)

            self.freeze_layers_encoder = [self.model.encoder.conv1,
                                          self.model.encoder.bn1,
                                          self.model.encoder.relu,
                                          self.model.encoder.maxpool,
                                          self.model.encoder.layer1,
                                          self.model.encoder.layer2,
                                          self.model.encoder.layer3]

            self.fine_tune_layers_encoder = list(self.model.encoder.layer4.parameters())

            self.freeze_layers_decoder = [self.model.decoder.layer1.block,
                                          self.model.decoder.layer2.block,
                                          self.model.decoder.layer3.block,
                                          self.model.decoder.layer4.block]

            self.fine_tune_layers_decoder = list(self.model.decoder.layer5.
                                                 parameters()) + list(self.model.decoder.final_conv.parameters())

        else:
            self.model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
                classes=len(self.classes),
                activation=self.activation,
            )
        return self.model

    def setup_model(self):
        # setup the model loss, metrics and optimizer
        self.model_loss = smp.utils.losses.BCEJaccardLoss(eps=1.)
        self.metrics = [
            smp.utils.metrics.IoUMetric(eps=1.),
            smp.utils.metrics.FscoreMetric(eps=1.),
        ]

        if self.fine_tuning:
            for layer in self.freeze_layers_encoder:
                layer.require_grad = False
            for layer in self.freeze_layers_decoder:
                layer.require_grad = False
            self.optimizer = torch.optim.Adam([
                {'params': self.fine_tune_layers_decoder, 'lr': 1e-4},
                {'params': self.fine_tune_layers_encoder, 'lr': 1e-6},
            ])
        else:
            self.model.decoder.layer1.require_grad = False
            self.optimizer = torch.optim.Adam([
                {'params': self.model.decoder.parameters(), 'lr': 1e-4},
                {'params': self.model.encoder.parameters(), 'lr': 1e-6},
            ])

    @staticmethod
    def get_training_augmentation_padding():
        # Add padding to make image shape divisible by 32
        test_transform = [
            albu.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, (0, 0, 0))
        ]
        return albu.Compose(test_transform)

    @staticmethod
    def get_training_augmentation_simple():
        # add padding and also a few simple augmentations
        test_transform = [
            albu.Rotate(limit=15, interpolation=1, border_mode=4, value=None, always_apply=False, p=0.5),
            albu.VerticalFlip(always_apply=False, p=0.5),
            albu.HorizontalFlip(always_apply=False, p=0.5),
            albu.Transpose(always_apply=False, p=0.5),
            albu.CenterCrop(height=200, width=200, always_apply=False, p=0.5),
            albu.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, (0, 0, 0))
        ]
        return albu.Compose(test_transform)

    def load_results(self, model_name=None):
        # load the results
        if model_name is not None:
            log_dir = self.root_dir + 'logs/' + self.loss + '/' + model_name
        else:
            log_dir = self.log_dir
        with open(log_dir + '/train_loss', 'rb') as f:
            train_loss = pickle.load(f)
        with open(log_dir + '/valid_loss', 'rb') as f:
            valid_loss = pickle.load(f)
        with open(log_dir + '/train_score', 'rb') as f:
            train_score = pickle.load(f)
        with open(log_dir + '/valid_score', 'rb') as f:
            valid_score = pickle.load(f)
        return train_loss, valid_loss, train_score, valid_score

    def write_results(self, train_loss, valid_loss, train_score, valid_score):
        with open(self.log_dir + '/train_loss', 'wb') as f:
            pickle.dump(train_loss, f)
        with open(self.log_dir + '/valid_loss', 'wb') as f:
            pickle.dump(valid_loss, f)
        with open(self.log_dir + '/train_score', 'wb') as f:
            pickle.dump(train_score, f)
        with open(self.log_dir + '/valid_score', 'wb') as f:
            pickle.dump(valid_score, f)

    def train_model(self):

        # train the model
        train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.model_loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.model_loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )

        max_score = 0

        train_loss = np.zeros(self.epochs_num)
        valid_loss = np.zeros(self.epochs_num)
        train_score = np.zeros(self.epochs_num)
        valid_score = np.zeros(self.epochs_num)

        valid_size = int(self.validation_ratio * len(self.full_dataset_pure))
        remaining_size = len(self.full_dataset_pure) - valid_size

        for i in range(0, self.epochs_num):

            # During every epoch randomly sample from the dataset, for training and validation dataset members
            train_dataset = self.full_dataset
            valid_dataset, remaining_dataset = torch.utils.data.random_split(self.full_dataset_pure,
                                                                             [valid_size, remaining_size])
            train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=12)
            valid_loader = DataLoader(valid_dataset, batch_size=24, drop_last=True)

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou'] or fine_tuning:
                max_score = valid_logs['iou']
                torch.save(self.model, self.model_dir)
                print('Model saved!')

            if i == 10:
                self.optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
            if i == 25:
                self.optimizer.param_groups[0]['lr'] = 1e-6
                print('Decrease decoder learning rate to 1e-6!')

            # get the loss logs
            train_loss[i] = train_logs['bce_jaccard_loss']
            valid_loss[i] = valid_logs['bce_jaccard_loss']
            train_score[i] = train_logs['f-score']
            valid_score[i] = valid_logs['f-score']

        # if continuing training, then load the previous loss and f-score logs
        if self.continue_train:
            train_loss_prev, valid_loss_prev, train_score_prev, valid_score_prev = self.load_results()
            train_loss = np.append(train_loss_prev, train_loss)
            valid_loss = np.append(valid_loss_prev, valid_loss)
            train_score = np.append(train_score_prev, train_score)
            valid_score = np.append(valid_score_prev, valid_score)
        self.write_results(train_loss, valid_loss, train_score, valid_score)

    def evaluate_model(self, scanner_cls=None):
        # load best saved checkpoint
        best_model = torch.load(self.model_dir)

        if scanner_cls is None:
            print("Testing for: all")
        else:
            print("Testing for: ", self.scanner_classes[scanner_cls])

        full_dataset_test = Dataset(
            self.x_dir_test,
            self.y_dir_test,
            classes=self.classes,
            augmentation=self.get_training_augmentation_padding(),
            scanner=scanner_cls
        )

        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=self.model_loss,
            metrics=self.metrics,
            device=self.device,
            verbose=False
        )

        test_size = int(len(full_dataset_test) * self.test_ratio)
        remaining_size = len(full_dataset_test) - test_size

        f_scores = []
        for i in range(20):
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset_test,
                                                                        [remaining_size, test_size])
            test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=1)
            logs = test_epoch.run(test_loader)
            f_scores.append(logs['f-score'])
        print("F-score Mean: ", np.mean(f_scores))
        print("F-scores Standard Dev: ", np.std(f_scores))
        print("F-scores: ", f_scores)
        print("*************************\n")

        return np.mean(f_scores)

    def plot_results(self, model_name=None):
        # load the results and make a plot
        if model_name is not None:
            plot_dir = self.root_dir + '/plots/' + self.loss + '/' + model_name + '.png'
        else:
            plot_dir = self.root_dir + '/plots/' + self.loss + '/' + self.model_name + '.png'

        x = np.arange(self.epochs_num)

        train_loss, valid_loss, train_score, valid_score = self.load_results(model_name)

        plt.plot(x, train_score)
        plt.plot(x, valid_score)
        plt.legend(['train_score', 'valid_score'], loc='lower right')
        plt.yticks(np.arange(0.0, 1.0, step=0.1))
        plt.savefig(plot_dir, bbox_inches='tight')
        plt.clf()

    def dump_results(self, results):
        file = os.path.join(self.root_dir, 'all_class_results.npy')
        data = np.load(file)

        print(data.shape, results.shape)
        data = np.vstack([data, results])
        print(data.shape)
        np.save(file, data)


if __name__ == "__main__":
    train = True
    continue_train = False
    test = True
    test_pure = False
    mode = "train_scanner"
    fine_tuning = False
    scanner_class = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    for config in configs:
        if config["mode"] in mode:
            config["model_name"] = config["model_name"] + '_{}'.format(scanner_class)
            unet_model = UnetTumorSegmentator(**config)
            unet_model.continue_train = continue_train
            unet_model.fine_tuning = fine_tuning
            unet_model.scanner_class = scanner_class
            unet_model.create_folders()
            unet_model.set_dataset_paths()
            unet_model.scanner_class_sizes()
            unet_model.create_dataset()
            unet_model.create_model()
            unet_model.setup_model()
            if train:
                unet_model.train_model()
            if test:
                pure_scores = []
                scores = []
                # for scanner_id, scanner in enumerate(unet_model.scanner_classes):
                """
                unet_model.model_name = config["model_name"] + '_{}'.format(scanner_class)
                unet_model.model_dir = unet_model.root_dir + 'models/' + \
                                       unet_model.loss + '/' + \
                                       unet_model.fold + '/' + \
                                       unet_model.model_name
                """
                scores.append(unet_model.evaluate_model(scanner_cls=None))
                """
                unet_model.model_name = 'model_epochs100_percent0_pure_vis_5'
                unet_model.model_dir = unet_model.root_dir + 'models/' + \
                                       unet_model.loss + '/' + \
                                       unet_model.fold + '/' + \
                                       unet_model.model_name
                pure_scores.append(unet_model.evaluate_model(scanner_cls=scanner_id))
                

                x = np.arange(len(scores))
                width = 0.35

                unet_model.dump_results(results=np.array(scores))

                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(111)
                ax.bar(x, scores, label="conditioned", width=width)
                # ax.bar(x + width, pure_scores, label="unconditioned", width=width)
                ax.set_title("F-scores of model conditioned on scanner classes")
                ax.set_ylim(0.0, 1.0)
                ax.set_xticks(x + width / 2)
                ax.set_yticks(np.linspace(0.0, 1.0, 50))
                ax.set_ylabel("f-score")
                ax.set_xlabel("Scanner Classes")
                ax.set_xticklabels(unet_model.scanner_classes)
                ax.legend()
                ax.grid()
                fig.tight_layout()
                plt.savefig(unet_model.scanner_plots_paths +
                            "new")
                plt.close(fig)
                """

            if test_pure:
                for scanner_id, scanner in enumerate(unet_model.scanner_classes):
                    unet_model.evaluate_model(scanner_cls=scanner_id)
