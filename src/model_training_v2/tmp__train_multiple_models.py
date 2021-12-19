import json
import logging
import os
from dataclasses import dataclass, field
from typing import List
import random
import copy

import cv2
import torch
import albumentations as A
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
import numpy as np
from torch import nn, optim


CPU_DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
print(f'DEVICE = {DEVICE}')


TILES_BASE_DIR = "/media/data/local/corn/processed_stride768_v2"
UNCROPPED_TILE_SIZE = (512 + 256)  # in pixels
CROPPED_TILE_SIZE = 512

# done
# TILES_BASE_DIR = "/media/data/local/corn/processed_stride384_v2"
# UNCROPPED_TILE_SIZE = (512 + 256) // 2  # in pixels
# CROPPED_TILE_SIZE = 512 // 2

# to do
# TILES_BASE_DIR = "/media/data/local/corn/processed_stride1152_v2"
# UNCROPPED_TILE_SIZE = (512 + 256) // 2 * 3  # in pixels
# CROPPED_TILE_SIZE = 512 // 2 * 3



SUBDIRECTORIES_TO_PROCESS_TRAIN = [
    "kukurydza_5_ha",
    "kukurydza_10_ha",
    "kukurydza_11_ha",
    "kukurydza_13_ha",
    "kukurydza_15_ha",
    "kukurydza_18_ha",
    "kukurydza_25_ha",
    "kukurydza_38_ha",
    "kukurydza_60_ha",
]


# if TEST or VALIDATIONS are empty, random part of training set will be used
SUBDIRECTORIES_TO_PROCESS_VALID = [
]

SUBDIRECTORIES_TO_PROCESS_TEST = [
]


# UNCROPPED_TILE_SIZE = (512 + 256)  # in pixels
# CROPPED_TILE_SIZE = 512


CROP_TILE_MARGIN = (UNCROPPED_TILE_SIZE - CROPPED_TILE_SIZE) // 2


@dataclass
class TilesPaths:
    img_paths: List = field(default_factory=lambda: [])
    mask_paths: List = field(default_factory=lambda: [])

    def split_into_train_valid_test(self, percentage_for_train=0.8):
        N = len(self.img_paths)
        test_percentage = (1 - percentage_for_train) / 2
        sp = [int(N * percentage_for_train), int(N * (1 - test_percentage))]  # dataset split points
        tile_paths_train = TilesPaths(img_paths=self.img_paths[:sp[0]], mask_paths=self.mask_paths[:sp[0]])
        tile_paths_valid = TilesPaths(img_paths=self.img_paths[sp[0]:sp[1]], mask_paths=self.mask_paths[sp[0]:sp[1]])
        tile_paths_test = TilesPaths(img_paths=self.img_paths[sp[1]:], mask_paths=self.mask_paths[sp[1]:])
        return tile_paths_train, tile_paths_valid, tile_paths_test

    def __add__(self, other):
        new = copy.deepcopy(self)
        new.img_paths += other.img_paths
        new.mask_paths += other.mask_paths
        return new

    def shuffle(self):
        c = list(zip(self.img_paths, self.mask_paths))
        random.shuffle(c)
        self.img_paths, self.mask_paths = zip(*c)


def get_tile_paths_for_directories_with_split(dir_names):
    tile_paths_train = TilesPaths()
    tile_paths_valid = TilesPaths()
    tile_paths_test = TilesPaths()
    for dir_name in dir_names:
        all_tile_paths = get_tile_paths_for_directories(dir_names=[dir_name])
        new_train, new_valid, new_test = all_tile_paths.split_into_train_valid_test()
        tile_paths_train += new_train
        tile_paths_valid += new_valid
        tile_paths_test += new_test

    tile_paths_train.shuffle()
    tile_paths_valid.shuffle()
    tile_paths_test.shuffle()

    return tile_paths_train, tile_paths_valid, tile_paths_test


def get_tile_paths_for_directories(dir_names, shuffle=True) -> TilesPaths:
    tile_paths = TilesPaths()
    for dir_name in dir_names:
        dir_path = os.path.join(TILES_BASE_DIR, dir_name)
        file_names = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

        mask_files_prefixes = set([f[:f.rfind('_')] for f in file_names if 'mask' in f])
        img_files_prefixes = set([f[:f.rfind('_')] for f in file_names if 'img' in f])
        common_files_prefixes = mask_files_prefixes.intersection(img_files_prefixes)
        all_files_prefixes = mask_files_prefixes.union(img_files_prefixes)
        missing_files_prefixes = all_files_prefixes - common_files_prefixes

        if missing_files_prefixes:
            raise Exception(
                f"Some files don't have correponding pair in mask/image: {missing_files_prefixes} in {dir_path}")

        common_files_prefixes = list(common_files_prefixes)
        if shuffle:
            random.shuffle(common_files_prefixes)
        for file_prefix in common_files_prefixes:
            img_file_name = file_prefix + '_img.png'
            mask_file_name = file_prefix + '_mask.png'
            tile_paths.img_paths.append(os.path.join(dir_path, img_file_name))
            tile_paths.mask_paths.append(os.path.join(dir_path, mask_file_name))
    return tile_paths


if SUBDIRECTORIES_TO_PROCESS_VALID and SUBDIRECTORIES_TO_PROCESS_TEST:
    # we have valid tiles for test/valid
    tile_paths_train = get_tile_paths_for_directories(SUBDIRECTORIES_TO_PROCESS_TRAIN)
    tile_paths_valid = get_tile_paths_for_directories(SUBDIRECTORIES_TO_PROCESS_VALID)
    tile_paths_test = get_tile_paths_for_directories(SUBDIRECTORIES_TO_PROCESS_TEST)
else:
    tile_paths_train, tile_paths_valid, tile_paths_test = get_tile_paths_for_directories_with_split(
        SUBDIRECTORIES_TO_PROCESS_TRAIN)

print(f'Number of tiles train = {len(tile_paths_train.img_paths)}')
print(f'Number of tiles validation = {len(tile_paths_valid.img_paths)}')
print(f'Number of tiles test = {len(tile_paths_test.img_paths)}')


SEGMENTATION_CLASS_VALUES = [0, 255, 127]
NUMBER_OF_SEGMENTATION_CLASSES = len(SEGMENTATION_CLASS_VALUES)


class CornFieldDamageDataset(torch.utils.data.Dataset):
    def __init__(self, img_file_paths, mask_file_paths, augment=True, ):
        self.img_file_paths = img_file_paths
        self.mask_file_paths = mask_file_paths
        assert (len(self.img_file_paths) == len(mask_file_paths))
        if augment:
            self._img_and_mask_transform = self._get_img_and_mask_augmentation_tranform()  # augmentation transform
        else:
            self._img_and_mask_transform = self._get_img_and_mask_crop_tranform()  # crop only transform

    def __len__(self):
        return len(self.mask_file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = cv2.imread(self.img_file_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # not really needed I guess
        mask = cv2.imread(self.mask_file_paths[idx], cv2.IMREAD_GRAYSCALE)

        transformed = self._img_and_mask_transform(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']

        masks = [(mask == v) for v in SEGMENTATION_CLASS_VALUES]
        mask_stacked = np.stack(masks, axis=0).astype('float')

        image = image.astype('float')
        image /= 255
        image = image.transpose(2, 0, 1)

        return image.astype('float32'), mask_stacked.astype('float32')

    def _get_img_and_mask_augmentation_tranform(self):
        # Declare an augmentation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomScale(scale_limit=0.15),  # above scale 0.16 images are too small
            A.Rotate(limit=90),  # degrees
            # TODO normalize instead divide by 255?
            A.Crop(x_min=CROP_TILE_MARGIN, y_min=CROP_TILE_MARGIN, x_max=UNCROPPED_TILE_SIZE - CROP_TILE_MARGIN,
                   y_max=UNCROPPED_TILE_SIZE - CROP_TILE_MARGIN),
            # TODO ToTensorV2 instead of manual stacking and transpoition?
        ])
        # TODO - color, contrast, gamma, randomShadow, rain
        return transform

    def _get_img_and_mask_crop_tranform(self):
        transform = A.Compose([
            A.Crop(x_min=CROP_TILE_MARGIN, y_min=CROP_TILE_MARGIN, x_max=UNCROPPED_TILE_SIZE - CROP_TILE_MARGIN,
                   y_max=UNCROPPED_TILE_SIZE - CROP_TILE_MARGIN),
        ])
        return transform


train_dataset = CornFieldDamageDataset(img_file_paths=tile_paths_train.img_paths,
                                       mask_file_paths=tile_paths_train.mask_paths)
valid_dataset = CornFieldDamageDataset(img_file_paths=tile_paths_valid.img_paths,
                                       mask_file_paths=tile_paths_valid.mask_paths, augment=False)
test_dataset = CornFieldDamageDataset(img_file_paths=tile_paths_test.img_paths,
                                      mask_file_paths=tile_paths_test.mask_paths, augment=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, drop_last=True)

architectures_encoders_weights_vec = [
    # {'architecture': smp.Unet, 'encoder': "efficientnet-b0", 'weights': 'imagenet'},
    # {'architecture': smp.UnetPlusPlus, 'encoder': "efficientnet-b0", 'weights': 'imagenet'},
    # {'architecture': smp.DeepLabV3, 'encoder': "efficientnet-b0", 'weights': 'i magenet'},
    # {'architecture': smp.DeepLabV3Plus, 'encoder': "efficientnet-b0", 'weights': 'imagenet'},
    # {'architecture': smp.PAN, 'encoder': "efficientnet-b0", 'weights': 'imagenet'},
    # {'architecture': smp.FPN, 'encoder': "efficientnet-b0", 'weights': 'imagenet'},
    # {'architecture': smp.MAnet, 'encoder': "efficientnet-b0", 'weights': 'imagenet'},
    # {'architecture': smp.Linknet, 'encoder': "efficientnet-b0", 'weights': 'imagenet'},
    #
    # {'architecture': smp.Unet, 'encoder': "efficientnet-b0", 'weights': 'imagenet'},
    # {'architecture': smp.Unet, 'encoder': "efficientnet-b3", 'weights': 'imagenet'},
    # {'architecture': smp.Unet, 'encoder': "resnet18", 'weights': 'imagenet'},
    # {'architecture': smp.Unet, 'encoder': "vgg11", 'weights': 'imagenet'},




    # {'architecture': smp.Unet, 'encoder': "efficientnet-b4", 'weights': 'imagenet'},
    # {'architecture': smp.Unet, 'encoder': "efficientnet-b1", 'weights': 'imagenet'},
    # {'architecture': smp.Unet, 'encoder': "efficientnet-b2", 'weights': 'imagenet'},
    # {'architecture': smp.Unet, 'encoder': "efficientnet-b5", 'weights': 'imagenet'},



    # {'architecture': smp.UnetPlusPlus, 'encoder': "efficientnet-b0", 'weights': 'imagenet'},  // done
    # {'architecture': smp.UnetPlusPlus, 'encoder': "efficientnet-b1", 'weights': 'imagenet'},
    # {'architecture': smp.UnetPlusPlus, 'encoder': "efficientnet-b2", 'weights': 'imagenet'},
    # {'architecture': smp.UnetPlusPlus, 'encoder': "efficientnet-b3", 'weights': 'imagenet'},
    # {'architecture': smp.UnetPlusPlus, 'encoder': "mobilenet_v2", 'weights': 'imagenet'},
    # {'architecture': smp.UnetPlusPlus, 'encoder': "densenet121", 'weights': 'imagenet'},
    # {'architecture': smp.UnetPlusPlus, 'encoder': "resnet18", 'weights': 'imagenet'},
    # {'architecture': smp.UnetPlusPlus, 'encoder': "se_resnet50", 'weights': 'imagenet'},
    # {'architecture': smp.UnetPlusPlus, 'encoder': "dpn68", 'weights': 'imagenet'},
    {'architecture': smp.UnetPlusPlus, 'encoder': "timm-resnest14d", 'weights': 'imagenet'},
    {'architecture': smp.UnetPlusPlus, 'encoder': "vgg11", 'weights': 'imagenet'},

]


for architectures_encoders_weights in architectures_encoders_weights_vec:
    try:
        results = {}
        results['architectures_encoders_weights'] = architectures_encoders_weights

        model = architectures_encoders_weights['architecture'](
            encoder_name=architectures_encoders_weights['encoder'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=architectures_encoders_weights['weights'],     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )

        # criterion = nn.CrossEntropyLoss()  # class imbalance is typically taken care of simply by assigning loss multipliers to each class,
        starting_learning_rate = 0.00001

        loss = smp.utils.losses.DiceLoss()
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5, name='IoU'),
            smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[1, 2], name='IoU-0'),
            smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0, 2], name='IoU-1'),
            smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0, 1], name='IoU-2'),
            smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[2]),
            smp.utils.metrics.Accuracy(threshold=0.5, ignore_channels=[2]),
            smp.utils.metrics.Recall(threshold=0.5, ignore_channels=[2]),
            smp.utils.metrics.Precision(threshold=0.5, ignore_channels=[2]),
        ]

        # optimizer = optim.SGD(model_fnn.parameters(), lr=0.0001, momentum=0.9)
        optimizer = torch.optim.Adam([
            #     dict(params=model.parameters(), lr=(0.0001)),  # 0.0001  #   0.000003 gives 80 epoch for 768 stride

            #     dict(params=model.parameters(), lr=0.000008),  # 0.0001  #   0.000003 gives 80 epoch for 768 stride
            dict(params=model.parameters(), lr=starting_learning_rate),  # 0.0001  #   0.000003 gives 80 epoch for 768 stride
        ])

        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True,
        )

        for e in [valid_epoch, train_epoch]:
            e.metrics[1].__name__ = "IoU_Class0"
            e.metrics[2].__name__ = "IoU_Class1"
            e.metrics[3].__name__ = "IoU_Class2"


        max_score = 0
        train_logs_vec = []
        valid_logs_vec = []
        best_model = None

        # epoch_to_decrease_learning_rate = 10

        lrs = [0.0005] * 3 + [0.0001] * 4 + [0.00005] * 9 + [0.00001] * 9 + [0.000005] * 9 + [0.000005] * 5
        number_of_epochs = len(lrs)
        results['lrs'] = lrs

        for i in range(0, number_of_epochs):
            optimizer.param_groups[0]['lr'] = lrs[i]
            print(f"Learning rate {optimizer.param_groups[0]['lr']}")

            print(f'\nEpoch: {i}')
            train_logs_vec.append(train_epoch.run(train_loader))
            valid_logs = valid_epoch.run(valid_loader)
            valid_logs_vec.append(valid_logs)

            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                model.to(CPU_DEVICE)
                best_model = copy.deepcopy(model)
                model.to(DEVICE)

        #     if i == epoch_to_decrease_learning_rate:
        #         optimizer.param_groups[0]['lr'] /= 2
        #         print(f"Decrease decoder learning rate to {optimizer.param_groups[0]['lr']}")

        results['train_logs_vec'] = train_logs_vec
        results['valid_logs_vec'] = valid_logs_vec

        model = best_model
        model.to(DEVICE)

        test_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True,
        )

        test_result = test_epoch.run(test_loader)
        print(f'test_result = {test_result}')
        results['test_result'] = test_result

        file_name = 'model_' + architectures_encoders_weights['architecture'].__name__ + \
                    '_' + architectures_encoders_weights['encoder'] + \
                    '_' + architectures_encoders_weights['weights']
        model_file_dir = TILES_BASE_DIR + '/models__051121/'
        model_file_path = os.path.join(model_file_dir, file_name)
        results['model_file_path'] = model_file_path
        logs_file_path = os.path.join(model_file_dir, file_name + '_log.json')

        os.makedirs(model_file_dir, exist_ok=True)

        results['architectures_encoders_weights']['architecture'] = results['architectures_encoders_weights']['architecture'].__name__
        with open(logs_file_path, 'w') as f:
            json.dump(results, f)

        model = model.to('cpu')
        torch.save(model.state_dict(), model_file_path)
    except:
        logging.exception('ERROR')
        continue
