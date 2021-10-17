import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()  # increase limit of pixels (2^30), before importing cv2
import cv2

import geopandas
import numpy as np

from tiles_generation import util
from tiles_generation.segmentation.model import FieldDamageSegmentationModel
import config
from area import FieldArea, DamageArea
from geo_tiff_image_wrapper import GeoTiffImageWrapper
from tile import Tile


def main():
    subdirectories = config.SUBDIRECTORIES_TO_PROCESS
    model = FieldDamageSegmentationModel(config.MODEL_PATH)

    for i, subdirectory_name in enumerate(subdirectories):
        print('*'*30)
        print(f'Processing subdirectory "{subdirectory_name}"... ({i+1}/{len(subdirectories)})')

        subdirectory_data_path = os.path.join(config.BASE_DATA_DIR_PATH, subdirectory_name)
        process_subdirectory(
            data_dir_path=subdirectory_data_path,
            model=model)

    print('')
    print('*'*30)
    print(f'All subdirectories processed!')


def process_subdirectory(data_dir_path, model):
    tif_wrapper = GeoTiffImageWrapper(file_path=os.path.join(data_dir_path, config.TIF_FILE_NAME))
    field_area = FieldArea(file_path=os.path.join(data_dir_path, config.FIELD_AREA_FILE_NAME))
    field_area.create_mask_for_tif(tif_wrapper, show=False)
    tif_wrapper.img = field_area.apply_mask_on_img(tif_wrapper.img)

    damage_area = DamageArea(file_path=os.path.join(data_dir_path, config.DAMAGE_AREA_FILE_NAME))
    damage_area.create_mask_for_tif_and_area(
        tif_wrapper=tif_wrapper,
        field_area=field_area,
        point_damage_file_path=os.path.join(data_dir_path, config.POINT_DAMAGE_AREA_FILE_NAME),
        show=False)

    # TODO - change stride to samler value. Do not use border of tile to set results
    # or maybe just add smole overlap, and add binary or during setting the result

    x_bins_number = (tif_wrapper.img_size_x_pixels - config.TILE_SIZE) // config.TILE_SIZE + 1
    y_bins_number = (tif_wrapper.img_size_y_pixels - config.TILE_SIZE) // config.TILE_SIZE + 1
    total_tiles = x_bins_number * y_bins_number

    full_predicted_img = np.zeros(tif_wrapper.img.shape[:2], np.uint8)

    for y_bin_number in range(y_bins_number):
        for x_bin_number in range(x_bins_number):
            tile = Tile(x_bin_number=x_bin_number, y_bin_number=y_bin_number, stride=config.TILE_SIZE)
            assert tile.end_pixel_x < tif_wrapper.img_size_x_pixels
            assert tile.end_pixel_y < tif_wrapper.img_size_y_pixels
            is_rectangle_within_field = field_area.is_rectangle_within_field(tile=tile)

            ascii_char_to_draw = 'x' if is_rectangle_within_field else '_'
            print(ascii_char_to_draw, end='')
            tile_no = y_bin_number * x_bins_number + x_bin_number
            if x_bin_number == (x_bins_number - 1):
                print(f" Processing tile {tile_no} / {total_tiles} [{tile_no / total_tiles * 100:.2f}%]")

            if not is_rectangle_within_field:  # TODO remove
                continue

            tile_img_bgr = tile.get_field_roi_img(tif_wrapper.img)
            tile_img_rgb = cv2.cvtColor(tile_img_bgr, cv2.COLOR_BGR2RGB)
            # tile_img_rgb = tile_img_bgr

            # util.show_small_img(damage_area.mask_img[tile.roi_slice], 'damage_img[self.roi_slice]')
            predicted_mask_binary = model.predict_damage(tile_img_rgb, show=False)

            tile.set_mask_on_full_img(full_img=full_predicted_img, roi_img=predicted_mask_binary)

    # util.show_small_img(full_predicted_img, 'full_predicted_img')
    # cv2.waitKey()

    full_predicted_img = field_area.apply_mask_on_img(full_predicted_img)
    geo_data_frame = get_geo_data_frame(full_predicted_img=full_predicted_img, tif_wrapper=tif_wrapper)
    geo_data_frame.to_file(os.path.join(data_dir_path, config.DAMAGE_AREA_FROM_NN_FILE_NAME), driver="GPKG")


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


