from dataclasses import dataclass
from typing import Optional

import torch
import cv2
import albumentations as A
import numpy as np

import dataset_preparation
from dataset_preparation import DEFAULT_DATASET_SPLIT_FILE_NAME


@dataclass
class TileDimensions:
    uncropped_tile_size: int
    cropped_tile_size: int
    crop_tile_margin: int


SEGMENTATION_CLASS_VALUES = [0, 255, 127]
NUMBER_OF_SEGMENTATION_CLASSES = len(SEGMENTATION_CLASS_VALUES)


class CornFieldDamageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_file_paths,
                 mask_file_paths,
                 tile_dimenstion: TileDimensions,
                 mask_scalling: Optional[float] = None,
                 augment=True):
        self.img_file_paths = img_file_paths
        self.mask_file_paths = mask_file_paths
        self.tile_dimenstion = tile_dimenstion
        self.mask_scalling = mask_scalling

        self.cropping_params = {
            'x_min': self.tile_dimenstion.crop_tile_margin,
            'y_min': self.tile_dimenstion.crop_tile_margin,
            'x_max': self.tile_dimenstion.uncropped_tile_size - self.tile_dimenstion.crop_tile_margin,
            'y_max': self.tile_dimenstion.uncropped_tile_size - self.tile_dimenstion.crop_tile_margin
        }

        assert (len(self.img_file_paths) == len(mask_file_paths))
        if augment:
            self._img_and_mask_transform = self._get_img_and_mask_augmentation_tranform()  # augmentation transform
        else:
            self._img_and_mask_transform = self._get_img_and_mask_crop_tranform()  # crop only transform

    def __len__(self):
        return len(self.mask_file_paths)

    def img_path_at_index(self, idx):
        return self.img_file_paths[idx]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = cv2.imread(self.img_file_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # not really needed I guess
        mask = cv2.imread(self.mask_file_paths[idx], cv2.IMREAD_GRAYSCALE)

        transformed = self._img_and_mask_transform(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']

        if self.mask_scalling:  # resize output image
            mask = cv2.resize(mask,
                              dsize=(int(mask.shape[0] * self.mask_scalling), int(mask.shape[1] * self.mask_scalling)),
                              interpolation=cv2.INTER_NEAREST)

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
            A.Crop(**self.cropping_params),
            # TODO ToTensorV2 instead of manual stacking and transpoition?
        ])
        # TODO - color, contrast, gamma, randomShadow, rain
        return transform

    def _get_img_and_mask_crop_tranform(self):
        transform = A.Compose([
            A.Crop(**self.cropping_params),
        ])
        return transform


def get_train_valid_test_loaders(base_dir_path, batch_size, mask_scalling, dataset_name=DEFAULT_DATASET_SPLIT_FILE_NAME):
    tile_paths_train, tile_paths_valid, tile_paths_test = dataset_preparation.load_tiles_dataset_split(
        base_dir_path=base_dir_path,
        dataset_name=dataset_name)

    uncropped_tile_size = int(
        base_dir_path[base_dir_path.rfind('stride_') + len('stride_'):].replace('/', '').replace('\\', ''))
    cropped_tile_size = uncropped_tile_size * 2 // 3
    tile_dimenstion = TileDimensions(
        uncropped_tile_size=uncropped_tile_size,
        cropped_tile_size=cropped_tile_size,
        crop_tile_margin=(uncropped_tile_size - cropped_tile_size) // 2,
    )

    print(f'cropped_tile_size = {cropped_tile_size}')

    train_dataset = CornFieldDamageDataset(img_file_paths=tile_paths_train.get_img_paths(),
                                           mask_file_paths=tile_paths_train.get_mask_paths(),
                                           mask_scalling=mask_scalling,
                                           tile_dimenstion=tile_dimenstion)
    valid_dataset = CornFieldDamageDataset(img_file_paths=tile_paths_valid.get_img_paths(),
                                           mask_file_paths=tile_paths_valid.get_mask_paths(),
                                           mask_scalling=mask_scalling,
                                           tile_dimenstion=tile_dimenstion,
                                           augment=False)
    test_dataset = CornFieldDamageDataset(img_file_paths=tile_paths_test.get_img_paths(),
                                          mask_file_paths=tile_paths_test.get_mask_paths(),
                                          mask_scalling=mask_scalling,
                                          tile_dimenstion=tile_dimenstion,
                                          augment=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print(f'len(test_loader) = {len(test_loader)}')
    return train_loader, valid_loader, test_loader


def main():
    get_train_valid_test_loaders(base_dir_path='/media/data/local/corn/new/tiles_stride_768/', batch_size=1, mask_scalling=None)


if __name__ == '__main__':
    main()
