import os
import shutil

from tiles_generation.common.geo_tiff_image_wrapper import GeoTiffImageWrapperNDVI

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()  # increase limit of pixels (2^30), before importing cv2
import cv2
import git

from common import config
from common.area import FieldArea, DamageArea
from common.geo_tiff_image_wrapper import GeoTiffImageWrapper
from common.tile import Tile


def process_subdirectory(data_dir_path, output_dir_path):
    tif_wrapper = GeoTiffImageWrapper(file_path=os.path.join(data_dir_path, config.TIF_FILE_NAME))
    field_area = FieldArea(file_path=os.path.join(data_dir_path, config.FIELD_AREA_FILE_NAME))
    damage_area = DamageArea(file_path=os.path.join(data_dir_path, config.DAMAGE_AREA_FILE_NAME))

    field_area.create_mask_for_tif(tif_wrapper, show=False)
    damage_area.create_mask_for_tif_and_area(
        tif_wrapper=tif_wrapper,
        field_area=field_area,
        point_damage_file_path=os.path.join(data_dir_path, config.POINT_DAMAGE_AREA_FILE_NAME),
        show=False)

    tif_wrapper.apply_mask(field_area.mask_img)

    if 'NDVI' in data_dir_path:
        ndvi_wrapper = GeoTiffImageWrapperNDVI(file_path=os.path.join(data_dir_path, config.NVDI_FILE_PATH),
                                           number_of_channels=1)
        ndvi_wrapper.transform_to_rgb_tif_wrapper_pixels_space(tif_wrapper)
        ndvi_wrapper.apply_mask(field_area.mask_img)
    else:
        ndvi_wrapper = None

    x_bins_number = (tif_wrapper.img_size_x_pixels - config.TILE_SIZE) // config.TILE_STRIDE + 1
    y_bins_number = (tif_wrapper.img_size_y_pixels - config.TILE_SIZE) // config.TILE_STRIDE + 1
    total_tiles = x_bins_number * y_bins_number

    for y_bin_number in range(y_bins_number):
        for x_bin_number in range(x_bins_number):
            tile = Tile(x_bin_number=x_bin_number, y_bin_number=y_bin_number, stride=config.TILE_STRIDE)
            assert tile.end_pixel_x < tif_wrapper.img_size_x_pixels
            assert tile.end_pixel_y < tif_wrapper.img_size_y_pixels
            is_rectangle_within_field = field_area.is_rectangle_within_field(tile=tile)

            ascii_char_to_draw = 'x' if is_rectangle_within_field else '_'
            print(ascii_char_to_draw, end='')
            tile_no = y_bin_number * x_bins_number + x_bin_number
            if x_bin_number == (x_bins_number - 1):
                print(f" Processing tile {tile_no} / {total_tiles} [{tile_no / total_tiles * 100:.2f}%]")

            if not is_rectangle_within_field:
                continue

            tile.save(damage_img=damage_area.mask_img,
                      field_img=tif_wrapper.img,
                      tile_output_dir=output_dir_path,
                      ndvi_wrapper=ndvi_wrapper,
                      )


def copy_config_file(output_dir_path):
    dst_flle_path = os.path.join(output_dir_path, 'config.py')
    shutil.copy2(src=config.CONFIG_FILE_PATH, dst=dst_flle_path)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(dst_flle_path, "a") as file_object:
        file_object.write(f"\n\n# commit_hash = {sha}")


def main():
    subdirectories = config.SUBDIRECTORIES_TO_PROCESS
    for i, subdirectory_name in enumerate(subdirectories):
        print('*'*30)
        print(f'Processing subdirectory "{subdirectory_name}"... ({i+1}/{len(subdirectories)})')

        subdirectory_data_path = os.path.join(config.BASE_DATA_DIR_PATH, subdirectory_name)
        output_dir_path = os.path.join(config.BASE_OUTPUT_DATA_DIR_PATH, subdirectory_name)
        os.makedirs(output_dir_path, exist_ok=True)
        copy_config_file(output_dir_path)

        process_subdirectory(
            data_dir_path=subdirectory_data_path,
            output_dir_path=output_dir_path)

    print('')
    print('*'*30)
    print(f'All subdirectories processed! Results saved in "{config.BASE_OUTPUT_DATA_DIR_PATH}"')


if __name__ == '__main__':
    main()
    cv2.waitKey()
    print('Done! Bye!')


