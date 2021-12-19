import copy
import json
from dataclasses import dataclass, field
import os
import random
from typing import List


DEFAULT_DATASET_SPLIT_FILE_NAME = 'dataset_split.json'


@dataclass
class TilesPaths:
    base_paths: List = field(default_factory=lambda: [])  # subdirectory with core name
    base_dir_path: str = ''

    def split_into_train_valid_test(self, percentage_for_train=0.8):
        N = len(self.base_paths)
        test_percentage = (1 - percentage_for_train) / 2
        sp = [int(N * percentage_for_train), int(N * (1 - test_percentage))]  # dataset split points
        tile_paths_train = TilesPaths(self.base_paths[:sp[0]])
        tile_paths_valid = TilesPaths(self.base_paths[sp[0]:sp[1]])
        tile_paths_test = TilesPaths(self.base_paths[sp[1]:])
        return tile_paths_train, tile_paths_valid, tile_paths_test

    def __add__(self, other):
        new = copy.deepcopy(self)
        new.base_paths += other.base_paths
        return new

    def shuffle(self):
        random.shuffle(self.base_paths)

    def get_img_paths(self):
        img_paths = []
        for base_path in self.base_paths:
            img_paths.append(os.path.join(self.base_dir_path, base_path + '_img.png'))
        return img_paths

    def get_mask_paths(self):
        img_paths = []
        for base_path in self.base_paths:
            img_paths.append(os.path.join(self.base_dir_path, base_path + '_mask.png'))
        return img_paths



def get_tile_paths_for_directories_with_split(base_dir_path, dir_names):
    tile_paths_train = TilesPaths()
    tile_paths_valid = TilesPaths()
    tile_paths_test = TilesPaths()
    for dir_name in dir_names:
        all_tile_paths = get_tile_paths_for_directories(base_dir_path=base_dir_path, dir_names=[dir_name])
        new_train, new_valid, new_test = all_tile_paths.split_into_train_valid_test()
        tile_paths_train += new_train
        tile_paths_valid += new_valid
        tile_paths_test += new_test

    tile_paths_train.shuffle()
    tile_paths_valid.shuffle()
    tile_paths_test.shuffle()

    return tile_paths_train, tile_paths_valid, tile_paths_test


def get_tile_paths_for_directories(base_dir_path, dir_names, shuffle=True) -> TilesPaths:
    tile_paths = TilesPaths()
    for dir_name in dir_names:
        dir_path = os.path.join(base_dir_path, dir_name)
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
            tile_paths.base_paths.append(os.path.join(dir_name, file_prefix))
    return tile_paths


def split_tiles_into_train_valid_test_and_save(base_dir_path, subdirectories):
    tile_paths_train, tile_paths_valid, tile_paths_test = get_tile_paths_for_directories_with_split(
        base_dir_path=base_dir_path, dir_names=subdirectories)

    print(f'Number of tiles train = {len(tile_paths_train.get_img_paths())}')
    print(f'Number of tiles validation = {len(tile_paths_valid.get_img_paths())}')
    print(f'Number of tiles test = {len(tile_paths_test.get_img_paths())}')

    json_data = {
        'tile_paths_train': tile_paths_train.base_paths,
        'tile_paths_valid': tile_paths_valid.base_paths,
        'tile_paths_test': tile_paths_test.base_paths,
    }

    output_file_path = os.path.join(base_dir_path, DEFAULT_DATASET_SPLIT_FILE_NAME)
    with open(output_file_path, 'wt') as file:
        json.dump(json_data, file, indent=1)


def load_tiles_dataset_split(base_dir_path, dataset_name):
    input_file_path = os.path.join(base_dir_path, dataset_name)
    with open(input_file_path, 'r') as file:
        data = json.load(file)
    tile_paths_train = TilesPaths(base_dir_path=base_dir_path, base_paths=data['tile_paths_train'])
    tile_paths_valid = TilesPaths(base_dir_path=base_dir_path, base_paths=data['tile_paths_valid'])
    tile_paths_test = TilesPaths(base_dir_path=base_dir_path, base_paths=data['tile_paths_test'])

    return tile_paths_train, tile_paths_valid, tile_paths_test


def main():
    # split_tiles_into_train_valid_test_and_save(
    #     base_dir_path='/media/data/local/corn/new/tiles_stride_768/',
    #     subdirectories=[
    #         "Kukurydza_RGB_25_ha",
    #         "Kukurydza_RGB_25.5_ha",
    #         "Kukurydza_RGB_9_ha",
    #         'Kukurydza_RGB_66_ha',
    #
    #         "kukurydza_5_ha",
    #         "kukurydza_10_ha",
    #         "kukurydza_11_ha",
    #         "kukurydza_13_ha",
    #         "kukurydza_15_ha",
    #         "kukurydza_18_ha",
    #         "kukurydza_25_ha",
    #         "kukurydza_38_ha",
    #         "kukurydza_60_ha",
    #     ],
    # )

    # split_tiles_into_train_valid_test_and_save(
    #     base_dir_path='/media/data/local/corn/new/tiles_stride_768_ndvi/',
    #     subdirectories=[
    #         'Kukurydza_RGB_NDVI_50_ha'
    #     ],
    # )

    tile_paths_train, tile_paths_valid, tile_paths_test = load_tiles_dataset_split(
        base_dir_path='/media/data/local/corn/new/tiles_stride_768_ndvi/')
    print(tile_paths_train.get_img_paths())


if __name__ == '__main__':
    main()
