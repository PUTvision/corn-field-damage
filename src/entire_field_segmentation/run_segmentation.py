import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()  # increase limit of pixels (2^30), before importing cv2
import cv2

import geopandas
import numpy as np

from entire_field_segmentation import field_segmentation_config
from entire_field_segmentation.model import FieldDamageSegmentationModel

from tiles_generation.common import util
from tiles_generation.common.area import FieldArea, DamageArea
from tiles_generation.common.geo_tiff_image_wrapper import GeoTiffImageWrapper
from tiles_generation.common.tile import Tile


def main():
    subdirectories = field_segmentation_config.SUBDIRECTORIES_TO_PROCESS
    model = FieldDamageSegmentationModel(
        field_segmentation_config.MODEL_PATH,
        model_type=field_segmentation_config.MODEL_TYPE)

    for i, subdirectory_name in enumerate(subdirectories):
        print('*'*30)
        print(f'Processing subdirectory "{subdirectory_name}"... ({i+1}/{len(subdirectories)})')

        subdirectory_data_path = os.path.join(field_segmentation_config.BASE_DATA_DIR_PATH, subdirectory_name)
        process_subdirectory(
            data_dir_path=subdirectory_data_path,
            model=model)

    print('')
    print('*'*30)
    print(f'All subdirectories processed!')


def process_subdirectory(data_dir_path, model):
    tif_wrapper = GeoTiffImageWrapper(file_path=os.path.join(data_dir_path, field_segmentation_config.TIF_FILE_NAME))
    field_area = FieldArea(file_path=os.path.join(data_dir_path, field_segmentation_config.FIELD_AREA_FILE_NAME))
    field_area.create_mask_for_tif(tif_wrapper, show=False)
    tif_wrapper.img = field_area.apply_mask_on_img(tif_wrapper.img)

    stride = int(field_segmentation_config.TILE_SIZE * 0.9)  # slightly overlap tiles, because segmentation on the edges is not reliable
    x_bins_number = (tif_wrapper.img_size_x_pixels - field_segmentation_config.TILE_SIZE) // stride + 1
    y_bins_number = (tif_wrapper.img_size_y_pixels - field_segmentation_config.TILE_SIZE) // stride + 1
    total_tiles = x_bins_number * y_bins_number

    full_predicted_img = np.zeros(tif_wrapper.img.shape[:2], np.uint8)

    for y_bin_number in range(y_bins_number):
        for x_bin_number in range(x_bins_number):
            tile = Tile(x_bin_number=x_bin_number, y_bin_number=y_bin_number, stride=stride)
            assert tile.end_pixel_x < tif_wrapper.img_size_x_pixels
            assert tile.end_pixel_y < tif_wrapper.img_size_y_pixels
            is_rectangle_within_field = field_area.is_rectangle_within_field(tile=tile)

            ascii_char_to_draw = 'x' if is_rectangle_within_field else '_'
            print(ascii_char_to_draw, end='')
            tile_no = y_bin_number * x_bins_number + x_bin_number
            if x_bin_number == (x_bins_number - 1):
                print(f" Processing tile {tile_no} / {total_tiles} [{tile_no / total_tiles * 100:.2f}%]")

            # if not is_rectangle_within_field:  # TODO remove
            #     continue
            # if y_bin_number % 30 != 0 or x_bin_number % 30 != 0:
            #     continue

            tile_img_rgb = tile.get_field_roi_img(tif_wrapper.img)  # already rgb!
            predicted_mask_binary = model.predict_damage(tile_img_rgb, show=False)

            tile.set_mask_on_full_img(full_img=full_predicted_img,
                                      roi_img=predicted_mask_binary,
                                      with_overlap=True)

    full_predicted_img = field_area.apply_mask_on_img(full_predicted_img)
    damaged_pixels = DamageArea.calculate_damaged_pixels_count(full_predicted_img)
    print(f'Total calculated damage area: {damaged_pixels * tif_wrapper.get_square_meters_per_pixel} m^2')

    del field_area
    geo_data_frame = get_geo_data_frame(full_predicted_img=full_predicted_img, tif_wrapper=tif_wrapper)
    geo_data_frame.to_file(os.path.join(data_dir_path, field_segmentation_config.DAMAGE_AREA_FROM_NN_FILE_NAME), driver="GPKG")


def get_geo_data_frame(full_predicted_img, tif_wrapper):
    full_predicted_img = cv2.morphologyEx(full_predicted_img, cv2.MORPH_CLOSE, (11, 11))
    full_predicted_img = cv2.morphologyEx(full_predicted_img, cv2.MORPH_OPEN, (11, 11))
    contours, hierarchy = cv2.findContours(full_predicted_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = tif_wrapper.transform_contours_yx_pixels_to_epsg(contours)
    geometry = util.convert_cv_contours_to_shapely_geometry(cv_contours=contours, hierarchy=hierarchy)
    geo_data_frame = geopandas.geodataframe.GeoDataFrame(geometry=geometry, crs='EPSG:32633')
    return geo_data_frame


if __name__ == '__main__':
    main()
    cv2.waitKey()
    print('Done! Bye!')


