from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from argparse import ArgumentParser
import numpy as np
from code.utils import *
from model.unet_model import Model

"""
Using a U-net architecture for segmentation of Tumor Modalities
"""


class Train:
    def __init__(self,
                 mode,
                 model_name,
                 pure_ratio,
                 synthetic_ratio,
                 test_ratio,
                 validation_ratio,
                 root_dir,
                 epochs,
                 epochs_elapsed,
                 device,
                 fold,
                 real_dir,
                 synthetic_dir,
                 test_dir,
                 train_mode,
                 dataset,
                 **kwargs):

        # the fold number to execute the code for
        self.fold = 'fold_' + str(fold)

        self.root_dir = root_dir
        self.real_dir = real_dir
        self.synthetic_dir = synthetic_dir
        self.test_dir = test_dir
        self.t1ce, self.t1, self.t2, self.flair, self.label = \
            kwargs['t1ce'], kwargs['t1'], kwargs['t2'], kwargs['flair'], kwargs['label']
        self.images, self.masks = kwargs['images'], kwargs['masks']

        # epochs
        self.epochs_num = epochs
        self.epochs_elapsed = epochs_elapsed
        self.total_epochs = 0 + self.epochs_num

        # what proportion of pure data to be used
        self.pure_ratio = pure_ratio

        # proportion of synthetic or augmented data to be used, depending on the mode, w.r.t the pure dataset size
        # can be set upto 2.0 i.e. 200%
        self.synthetic_ratio = synthetic_ratio

        # proportion of test data to be used
        self.test_ratio = test_ratio

        # test and validation ratios to use
        self.validation_ratio = validation_ratio

        # training
        self.mode = mode
        self.device = device
        self.train_mode = train_mode
        self.dataset = dataset

        # set paths
        self.model_name = model_name

        self.model_dir = os.path.join(self.root_dir, 'models', self.fold, self.model_name)

        # dataset paths
        self.x_dir = dict()
        self.y_dir = None
        self.x_dir_syn = dict()
        self.y_dir_syn = None
        self.x_dir_test = dict()
        self.y_dir_test = None

        # full dataset and the pure dataset
        self.full_dataset = None
        self.full_dataset_pure = None

        # model setup
        self.model_loss = smp.utils.losses.BCEJaccardLoss(eps=1.)
        self.metrics = [
            smp.utils.metrics.IoUMetric(eps=1.),
            smp.utils.metrics.FscoreMetric(eps=1.)
        ]

        if self.dataset == 'brats':
            # scanner classes of BRATS dataset
            self.scanner_classes = [
                '2013', 'CBICA', 'TCIA01', 'TCIA02', 'TCIA03', 'TCIA04', 'TCIA05', 'TCIA06', 'TCIA08', 'TMC',
            ]
            from dataset.brats_dataset import Dataset
            self.Dataset = Dataset
            self.classes = ['t_2', 't_1', 't_3']
            self.model, self.optimizer = Model(root_dir, model_name, fold, mode, kwargs['encoder'],
                                               kwargs['activation'], dataset, len(self.classes)).create_model()
        else:
            self.scanner_classes = [
                'melanoma', 'seborrheic keratosis', 'nevus'
            ]
            from dataset.isic_dataset import Dataset
            self.Dataset = Dataset
            self.classes = ['lesion']
            self.model, self.optimizer = Model(root_dir, model_name, fold, mode, kwargs['encoder'],
                                               kwargs['activation'], dataset, len(self.classes)).create_model()

    def create_folders(self):
        if not os.path.join(self.root_dir, 'models', self.fold):
            os.makedirs(self.model_dir)

    def set_paths(self, root):
        if self.dataset == 'brats':
            d = dict()
            d['t1ce'] = os.path.join(root, self.t1ce)
            d['flair'] = os.path.join(root, self.flair)
            d['t2'] = os.path.join(root, self.t2)
            d['t1'] = os.path.join(root, self.t1)
            y = os.path.join(root, self.label)
        else:
            d = os.path.join(root, self.images)
            y = os.path.join(root, self.masks)

        return d, y

    def set_dataset_paths(self):
        self.x_dir, self.y_dir = self.set_paths(self.real_dir)
        self.x_dir_syn, self.y_dir_syn = self.set_paths(self.synthetic_dir)
        self.x_dir_test, self.y_dir_test = self.set_paths(self.test_dir)

    def create_dataset(self):
        self.full_dataset_pure = self.Dataset(
            self.x_dir,
            self.y_dir,
            classes=self.classes,
            augmentation=get_training_augmentation_padding(self.dataset),
        )

        pure_size = int(len(self.full_dataset_pure) * self.pure_ratio)
        self.full_dataset_pure = torch.utils.data.Subset(self.full_dataset_pure, np.arange(pure_size))

        if self.train_mode == 'real_only':
            self.full_dataset = self.full_dataset_pure

        elif self.train_mode == 'syn_only':
            full_dataset_syn = self.Dataset(
                self.x_dir_syn,
                self.y_dir_syn,
                classes=self.classes,
                augmentation=get_training_augmentation_simple(self.dataset),
                synthesized=True
            )

            self.full_dataset = full_dataset_syn
            self.full_dataset_pure = self.full_dataset

        else:
            full_dataset_syn = self.Dataset(
                self.x_dir_syn,
                self.y_dir_syn,
                classes=self.classes,
                augmentation=get_training_augmentation_padding(self.dataset),
                synthesized=True
            )

            synthetic_size = int(len(full_dataset_syn) * self.synthetic_ratio)
            full_dataset_syn = torch.utils.data.Subset(full_dataset_syn, np.arange(synthetic_size))

            self.full_dataset = torch.utils.data.ConcatDataset((self.full_dataset_pure, full_dataset_syn))

        return self.full_dataset, self.full_dataset_pure

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
            if max_score < valid_logs['iou'] or self.mode == 'fine_tuning':
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

    def evaluate_model(self, scanner_cls=None):
        # load best saved checkpoint
        best_model = torch.load(self.model_dir)

        if scanner_cls is None:
            print("Testing for: all")
        else:
            print("Testing for: ", self.scanner_classes[scanner_cls])

        full_dataset_test = self.Dataset(
            self.x_dir_test,
            self.y_dir_test,
            classes=self.classes,
            augmentation=get_training_augmentation_padding(self.dataset),
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

        return np.mean(f_scores)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True, default=1, help='dataset fold number if applicable')
    parser.add_argument('--train_mode', type=str, choices=['real_only', 'syn_only', 'mixed'], default='mixed')
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--pure_ratio', type=float, default=1.0)
    parser.add_argument('--synthetic_ratio', type=float, default=1.0)
    parser.add_argument('--test_ratio', type=float, default=1.0)
    parser.add_argument('--validation_ratio', type=float, default=0.1, help='proportion of train set for validation')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--epochs_elapsed', type=int, default=0, help="if continue train, then epochs already elapsed")
    parser.add_argument('--root_dir', type=str, required=True, help="root dir where models, logs and plot are saved")
    parser.add_argument('--real_dir', type=str, required=True, help="root dir for the real data directory")
    parser.add_argument('--synthetic_dir', type=str, required=True, help="root dir for the synthetic data directory")
    parser.add_argument('--test_dir', type=str, required=True, help="root dir for the synthetic data directory")
    parser.add_argument('--t1ce', type=str, help="name for t1ce data directories", default="train_t1ce_img_full")
    parser.add_argument('--t2', type=str, help="name for t1ce data directories", default="train_t2_img_full")
    parser.add_argument('--t1', type=str, help="name for t1ce data directories", default="train_t1_img_full")
    parser.add_argument('--flair', type=str, help="name for t1ce data directories", default="train_flair_img_full")
    parser.add_argument('--label', type=str, help="name for labels data directories", default="train_label_full")
    parser.add_argument('--images', type=str, help="name for image data directories of isic", default="images")
    parser.add_argument('--masks', type=str, help="name for mask data directories of isic", default="segmentation_masks")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'fine_tune', 'test', 'continue_train'])
    parser.add_argument('--encoder', type=str, choices=['resnet34'], default='resnet34')
    parser.add_argument('--activation', type=str, choices=['softmax'], default='softmax')
    parser.add_argument('--test_class', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], default=None)
    parser.add_argument('--dataset', type=str, choices=['brats', 'isic'], default='brats')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    unet_model = Train(**args.__dict__)

    unet_model.create_folders()
    unet_model.set_dataset_paths()
    unet_model.create_dataset()

    if args.mode == 'test':
        unet_model.evaluate_model(scanner_cls=args.test_class)
    else:
        unet_model.train_model()
