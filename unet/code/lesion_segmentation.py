import os
import sys
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
import albumentations as albu
import pickle
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from lesion_dataset import Dataset

"""
Using a U-net architecture for segmentation of Tumor Modalities
"""


class UnetTumorSegmentator:
    def __init__(self, mode, model_name, pure_ratio, scanner_class):
        self.fold = 'fold_1'
        self.root_dir = '/home/qasima/segmentation_models.pytorch/isic2018/'
        self.data_dir = '/home/qasima/segmentation_models.pytorch/isic2018/data/' + self.fold + '/'

        self.lesion_class = scanner_class

        # epochs
        self.epochs_num = 100
        self.total_epochs = 0 + self.epochs_num
        self.continue_train = False

        # what proportion of pure data to be used
        self.pure_ratio = pure_ratio

        # proportion of test data to be used
        self.test_ratio = 1.0

        # test and validation ratios to use
        self.validation_ratio = 0.1

        # training
        # mode = pure, none, elastic, coregistration, augmented, none_only or augmented_coregistered
        self.mode = mode
        self.encoder = 'resnet34'
        self.encoder_weights = None
        self.device = 'cuda'
        self.activation = 'softmax'
        self.loss = 'cross_entropy'

        # classes to be trained upon
        self.classes = ['lesion']

        # paths
        self.model_name = model_name
        self.log_dir = self.root_dir + 'logs/' + self.loss + '/' + self.fold + '/' + self.model_name
        self.model_dir = self.root_dir + 'models/' + self.loss + '/' + self.fold + '/' + self.model_name
        self.result_dir = self.root_dir + 'results/' + self.loss + '/' + self.fold + '/' + self.model_name
        self.scanner_plots_paths = self.root_dir + "scanner_class_plots/" + self.fold + '/'

        # dataset paths
        self.x_dir = None
        self.y_dir = None
        self.x_dir_syn = None
        self.y_dir_syn = None
        self.x_dir_test = None
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
        self.lesion_classes = [
            'melanoma',
            'seborrheic keratosis',
            'nevus'
        ]
        self.lesion_classes_size = []

    def create_folders(self):
        # create folders for results and logs to be saved
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        # if not os.path.exists(self.model_dir):
        #    os.makedirs(self.model_dir)
        if not os.path.exists(self.scanner_plots_paths):
            os.makedirs(self.scanner_plots_paths)

    def set_dataset_paths(self):
        # pure dataset
        self.x_dir = os.path.join(self.data_dir, 'train/images')
        self.y_dir = os.path.join(self.data_dir, 'train/segmentation_masks')

        # syn dataset
        if self.lesion_class == "balanced":
            self.x_dir_syn = list()
            for i, cls in enumerate(self.lesion_classes):
                self.x_dir_syn.append(os.path.join(self.data_dir, 'train_syn/{}'.format(i)))

        elif self.lesion_class == "none":
            self.x_dir_syn = os.path.join(self.data_dir, 'train_syn_vanilla')
        else:
            self.x_dir_syn = os.path.join(self.data_dir, 'train_syn/{}'.format(self.lesion_class))
        self.y_dir_syn = os.path.join(self.data_dir, 'train/segmentation_masks')

        # test dataset
        self.x_dir_test = os.path.join(self.data_dir, 'test/images')
        self.y_dir_test = os.path.join(self.data_dir, 'test/segmentation_masks')

    def get_lesion_class_sizes(self):
        for i, cls in enumerate(self.lesion_classes):
            lesion_dataset = Dataset(
                self.x_dir,
                self.y_dir,
                augmentation=self.get_training_augmentation_padding(),
                lesion_cls=self.lesion_classes[i]
            )
            self.lesion_classes_size.append(len(lesion_dataset))

    def create_dataset(self):
        # create the pure dataset
        self.full_dataset_pure = Dataset(
            self.x_dir,
            self.y_dir,
            augmentation=self.get_training_augmentation_padding(),
        )

        pure_size = int(len(self.full_dataset_pure) * self.pure_ratio)
        self.full_dataset_pure = torch.utils.data.Subset(self.full_dataset_pure, np.arange(pure_size))

        if self.mode == 'pure':
            # mode is pure then full dataset is the pure dataset
            self.full_dataset = self.full_dataset_pure
        elif self.mode == 'mixed':
            if self.lesion_class == "balanced":
                datasets = []
                max_lesion_size = max(self.lesion_classes_size)

                for i, cls in enumerate(self.lesion_classes):
                    full_dataset_lesion_syn = Dataset(
                        self.x_dir_syn[i],
                        self.y_dir_syn,
                        synthesized=True,
                        augmentation=self.get_training_augmentation_padding(),
                    )
                    full_dataset_lesion_syn = torch.utils.data.Subset(full_dataset_lesion_syn,
                                                                      np.arange(max_lesion_size -
                                                                                self.lesion_classes_size[i]))
                    datasets.append(full_dataset_lesion_syn)

                full_dataset_syn = torch.utils.data.ConcatDataset(datasets)
            else:
                full_dataset_syn = Dataset(
                    self.x_dir_syn,
                    self.y_dir_syn,
                    synthesized=True,
                    augmentation=self.get_training_augmentation_padding(),
                )
            self.full_dataset = torch.utils.data.ConcatDataset((self.full_dataset_pure, full_dataset_syn))
        elif self.mode == "syn_only":
            datasets = []
            max_lesion_size = max(self.lesion_classes_size)
            if self.lesion_class == "none":
                full_dataset_syn = Dataset(
                    self.x_dir_syn,
                    self.y_dir_syn,
                    synthesized=True,
                    augmentation=self.get_training_augmentation_padding(),
                )
            else:
                for i, cls in enumerate(self.lesion_classes):
                    full_dataset_lesion_syn = Dataset(
                        self.x_dir_syn[i],
                        self.y_dir_syn,
                        synthesized=True,
                        augmentation=self.get_training_augmentation_padding(),
                    )
                    full_dataset_lesion_syn = torch.utils.data.Subset(full_dataset_lesion_syn,
                                                                      np.arange(max_lesion_size -
                                                                                self.lesion_classes_size[i]))
                    datasets.append(full_dataset_lesion_syn)

                full_dataset_syn = torch.utils.data.ConcatDataset(datasets)

            self.full_dataset = full_dataset_syn
            self.full_dataset_pure = full_dataset_syn

        return self.full_dataset, self.full_dataset_pure

    def create_model(self):
        # create or load the model
        if self.continue_train:
            self.model = torch.load(self.model_dir)
        else:
            self.model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
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

        self.optimizer = torch.optim.Adam([
            {'params': self.model.decoder.parameters(), 'lr': 1e-4},
            {'params': self.model.encoder.parameters(), 'lr': 1e-6},
        ])

    @staticmethod
    def get_training_augmentation_padding():
        # Add padding to make image shape divisible by 32
        test_transform = [
            albu.PadIfNeeded(256, 256, cv2.BORDER_CONSTANT, (0, 0, 0)),
            albu.Resize(256, 256),
            albu.Normalize()
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
            train_loader = DataLoader(train_dataset, batch_size=72, shuffle=True, num_workers=8)
            valid_loader = DataLoader(valid_dataset, batch_size=36, drop_last=True, num_workers=4)

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou']:
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

    def evaluate_model(self, lesion_cls=None):
        # load best saved checkpoint
        best_model = torch.load(self.model_dir)
        if lesion_cls is None:
            print("Testing for: all")
        else:
            print("Testing for: ", self.lesion_classes[lesion_cls])

        if lesion_cls is None:
            full_dataset_test = Dataset(
                self.x_dir_test,
                self.y_dir_test,
                augmentation=self.get_training_augmentation_padding()
            )
        else:
            full_dataset_test = Dataset(
                self.x_dir_test,
                self.y_dir_test,
                augmentation=self.get_training_augmentation_padding(),
                lesion_cls=self.lesion_classes[lesion_cls]
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
        for i in range(10):
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset_test,
                                                                            [remaining_size, test_size])
            test_loader = DataLoader(test_dataset, batch_size=15, shuffle=True, num_workers=1)
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
    train = False
    continue_train = False
    test = True
    test_pure = False

    lesion_class = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

    unet_model = UnetTumorSegmentator("syn_only", "model_epochs100_percent100_isic_syn_only_segmentator", 0.0, lesion_class)
    unet_model.continue_train = continue_train
    unet_model.create_folders()
    unet_model.set_dataset_paths()
    unet_model.get_lesion_class_sizes()
    unet_model.create_dataset()
    unet_model.create_model()
    unet_model.setup_model()
    if train:
        unet_model.train_model()
    if test:
        pure_scores = []
        scores = []
        """
        unet_model.model_name = "model_epochs100_percent100_isic_{}".format(lesion_class)
        unet_model.model_dir = unet_model.root_dir + 'models/' + \
                               unet_model.loss + '/' + \
                               unet_model.fold + '/' + \
                               unet_model.model_name
        """
        scores.append(unet_model.evaluate_model(lesion_cls=None))
        """
        unet_model.model_name = 'model_epochs100_percent100_isic_256'
        unet_model.model_dir = unet_model.root_dir + 'models/' + \
                               unet_model.loss + '/' + \
                               unet_model.fold + '/' + \
                               unet_model.model_name
        pure_scores.append(unet_model.evaluate_model(lesion_cls=lesion_id))
        """

        exit()

        x = np.arange(len(scores))
        width = 0.35

        # unet_model.dump_results(np.array(scores))

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
        ax.set_xticklabels(unet_model.lesion_classes)
        ax.legend()
        ax.grid()
        fig.tight_layout()
        plt.savefig(unet_model.scanner_plots_paths +
                    "new")
        plt.close(fig)
    if test_pure:
        unet_model.evaluate_model()
