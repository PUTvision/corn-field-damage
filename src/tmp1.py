import os
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np
from geotiff import GeoTiff
import geopandas

from src import config, util


def show_small_img(img, name='img', show=False):
    if not show:
        return

    small_height = 500
    shape = img.shape[:2]
    img_small = cv2.resize(img, (small_height, int(512 / shape[1] * shape[0])))
    cv2.imshow(name, img_small)
    cv2.waitKey()


class Tile:
    def __init__(self, x_bin_number, y_bin_number):
        self.start_pixel_x = x_bin_number * config.TILE_STRIDE
        self.start_pixel_y = y_bin_number * config.TILE_STRIDE
        self.size = config.TILE_SIZE
        self.end_pixel_x = self.start_pixel_x + self.size - 1
        self.end_pixel_y = self.start_pixel_y + self.size - 1
        self.x_bin_number = x_bin_number
        self.y_bin_number = y_bin_number
        self.roi_slice = np.s_[self.start_pixel_y:self.end_pixel_y + 1, self.start_pixel_x:self.end_pixel_x + 1]

    def get_corners(self):
        return [
            (self.start_pixel_y, self.start_pixel_x),
            (self.start_pixel_y, self.end_pixel_x),
            (self.end_pixel_y, self.start_pixel_x),
            (self.end_pixel_y, self.end_pixel_x),
        ]

    def get_pixel_area(self):
        return self.size * self.size

    def save(self, damage_img, field_img, tile_output_dir):
        damage_roi = damage_img[self.roi_slice]
        field_roi = field_img[self.roi_slice]

        tile_img_file_name = f'tile_{self.x_bin_number:03d}_{self.y_bin_number:03d}_img.png'
        tile_mask_file_name = f'tile_{self.x_bin_number:03d}_{self.y_bin_number:03d}_mask.png'
        tile_img_file_path = os.path.join(tile_output_dir, tile_img_file_name)
        tile_mask_file_path = os.path.join(tile_output_dir, tile_mask_file_name)

        cv2.imwrite(img=damage_roi, filename=tile_img_file_path)
        cv2.imwrite(img=field_roi, filename=tile_mask_file_path)


@dataclass
class CoordinatesRect:
    """
    x coordinate related with longitude (horizontal position on map)
    y coordinate related with latitude (vertical position on map)

    Coordinates are increasing in top left direction
    """
    x_left: float
    y_upper: float
    x_right: float
    y_bottom: float

    @classmethod
    def from_2_tuples(cls, points):
        return cls(*points[0], *points[1])

    def get_x_distance_in_meters(self):
        return self.x_right - self.x_left

    def get_y_distance_in_meters(self):
        return self.y_upper - self.y_bottom


class GeoTiffImageWrapper:
    def __init__(self, tif_file_path: str):
        self.geo_tiff = GeoTiff(tif_file_path)
        assert self.geo_tiff.crs_code == 32633  # Processing supports only EPSG32633
        self.coord_bounding_box = CoordinatesRect.from_2_tuples(self.geo_tiff.tif_bBox)

        self.img = cv2.imread(tif_file_path)
        self.img_size_y_pixels, self.img_size_x_pixels, _ = self.img.shape

        self._pixels_per_epsg_x = 1 / self.spacial_resolution_x_in_meters  # so the same as pixels for one meter in x
        self._pixels_per_epsg_y = 1 / self.spacial_resolution_y_in_meters  # so the same as pixels for one meter in y

    def print_stats(self):
        print(f'tif EPSG32633 bounding box = {self.coord_bounding_box}')
        print(f'TIF image size xy in pixels: {self.img_size_y_pixels, self.img_size_x_pixels}')
        print(f'TIF image size xy in meters: '
              f'{self.coord_bounding_box.get_x_distance_in_meters():.3f}, '
              f'{self.coord_bounding_box.get_y_distance_in_meters():.3f}')
        print(f'TIF image resolution xy [meters per pixel]: '
              f'{self.spacial_resolution_x_in_meters:.3f}, {self.spacial_resolution_y_in_meters:.3f}')

    @property
    def spacial_resolution_x_in_meters(self) -> float:
        return self.coord_bounding_box.get_x_distance_in_meters() / (self.img_size_x_pixels - 1)

    @property
    def spacial_resolution_y_in_meters(self) -> float:
        return self.coord_bounding_box.get_y_distance_in_meters() / (self.img_size_y_pixels - 1)

    @property
    def get_square_meters_per_pixel(self) -> float:
        pixel_area_square_meters = self.spacial_resolution_x_in_meters * self.spacial_resolution_y_in_meters
        return pixel_area_square_meters

    def transform_polygons_to_xy_pixels(self, polygons: Tuple[np.ndarray, np.ndarray]) -> List[np.ndarray]:
        """
        Transform coordinates polygons to pixels contours (with cv2 format)
        :param polygons: List of tuples with two lists each (x and y points respoectively)
        :return: 2D contours list
        """
        yx_pixel_contours = []
        for polygon in polygons:
            x_epsg, y_epsg = polygon
            yx_px = np.zeros((len(x_epsg), 2), dtype=int)
            x_left = self.coord_bounding_box.x_left
            y_upper = self.coord_bounding_box.y_upper
            for i in range(len(x_epsg)):
                yx_px[i][0] = round((x_epsg[i] - x_left) * self._pixels_per_epsg_x)
                yx_px[i][1] = round((y_upper - y_epsg[i]) * self._pixels_per_epsg_y)
            yx_pixel_contours.append(yx_px)
        return yx_pixel_contours

    def apply_mask(self, mask_img, show=False):
        self.img = cv2.bitwise_and(self.img, self.img, mask=mask_img)
        util.show_small_img(img=self.img, name='field_masked', show=show)


class BaseArea:
    def __init__(self, file_path):
        self.data = geopandas.read_file(file_path)
        self.mask_img = None

    def _get_polygons_xy(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def create_mask_for_tif(self, tif_wrapper: GeoTiffImageWrapper, erode_contour_size: int = 0, show=False):
        shape = tif_wrapper.img.shape[:2]
        self.mask_img = np.zeros(shape, np.uint8)
        polygons = self._get_polygons_xy()
        yx_pixel_contours = tif_wrapper.transform_polygons_to_xy_pixels(polygons)
        cv2.fillPoly(self.mask_img, pts=yx_pixel_contours, color=255)

        if erode_contour_size:
            # erosion because field area goes over the real border of the field sometime
            self.mask_img = cv2.erode(self.mask_img, np.ones((erode_contour_size, erode_contour_size), np.uint8))

        util.show_small_img(img=self.mask_img, name='area_small', show=show)


class FieldArea(BaseArea):
    """ Wraps gpkg file with entire filed area mask (obszar.gpkg) """
    def __init__(self, file_path):
        super().__init__(file_path)

    def _get_polygons_xy(self):
        area_polygon = self.data.to_numpy()[0][0][0]
        x_epsg, y_epsg = area_polygon.exterior.coords.xy
        return [(x_epsg, y_epsg)]

    def create_mask_for_tif(self, tif_wrapper: GeoTiffImageWrapper, erode_contour_size: int = 0, show=False):
        erode_contour_size = config.FIELD_BORDER_EROSION_SIZE_PIXELS
        super().create_mask_for_tif(tif_wrapper=tif_wrapper, erode_contour_size=erode_contour_size, show=show)

    def is_rectangle_within_field(self, tile: Tile):
        corners_within_field = 0
        for corner in tile.get_corners():
            if self.mask_img[corner]:
                corners_within_field += 1
        if corners_within_field == 0:
            return False
        if corners_within_field == 4:
            return True

        minimum_coverage_percentage = config.MINIMUM_TILE_FIELD_COVERAGE_PERCENTAGE
        mask_roi = self.mask_img[tile.roi_slice]
        coverage_percentage = cv2.countNonZero(mask_roi) / tile.get_pixel_area() * 100
        if coverage_percentage >= minimum_coverage_percentage:
            return True

        return False


class DamageArea(BaseArea):
    """ Wraps gpkg file with damage filed area mask (szkody_placowe.gpkg) """
    def __init__(self, file_path):
        super().__init__(file_path)

    def _get_polygons_xy(self):
        polygons = []
        for shapely_polygon in self.data.to_numpy():
            area_polygon = shapely_polygon[0][0]
            x_epsg, y_epsg = area_polygon.exterior.coords.xy
            polygons.append((x_epsg, y_epsg))
        return polygons

    def create_mask_for_tif_and_area(self, tif_wrapper: GeoTiffImageWrapper, field_area: FieldArea, show=False):
        super().create_mask_for_tif(tif_wrapper=tif_wrapper, show=show)
        # field_area_mask_img_copy = field_area.mask_img.copy()
        self.mask_img[np.where(field_area.mask_img == 0)] = 88
        show_small_img(self.mask_img, name='damage', show=show)


def main():
    tif_file_path = os.path.join(config.BASE_DATA_DIR_PATH, 'fotomapa.tif')
    field_area_file_path = os.path.join(config.BASE_DATA_DIR_PATH, 'obszar.gpkg')
    damage_area_file_path = os.path.join(config.BASE_DATA_DIR_PATH, 'szkody_placowe.gpkg')

    tif_wrapper = GeoTiffImageWrapper(tif_file_path=tif_file_path)
    tif_wrapper.print_stats()

    field_area = FieldArea(field_area_file_path)
    damage_area = DamageArea(damage_area_file_path)

    field_area.create_mask_for_tif(tif_wrapper, show=False)
    damage_area.create_mask_for_tif_and_area(tif_wrapper, field_area=field_area, show=False)

    tif_wrapper.apply_mask(field_area.mask_img)

    x_bins_number = (tif_wrapper.img_size_x_pixels - config.TILE_SIZE) // config.TILE_STRIDE + 1
    y_bins_number = (tif_wrapper.img_size_y_pixels - config.TILE_SIZE) // config.TILE_STRIDE + 1
    total_tiles = x_bins_number * y_bins_number

    os.makedirs(config.BASE_OUTPUT_DATA_DIR_PATH, exist_ok=True)

    for y_bin_number in range(y_bins_number):
        for x_bin_number in range(x_bins_number):
            tile_no = y_bin_number * x_bins_number + x_bin_number
            if x_bin_number == (x_bins_number - 1):
                print(f" Processing tile {tile_no} / {total_tiles} [{tile_no/total_tiles*100:.2f}%]")

            tile = Tile(x_bin_number=x_bin_number, y_bin_number=y_bin_number)
            assert tile.end_pixel_x < tif_wrapper.img_size_x_pixels
            assert tile.end_pixel_y < tif_wrapper.img_size_y_pixels

            is_rectangle_within_field = field_area.is_rectangle_within_field(tile=tile)
            ascii_char_to_draw = 'x' if is_rectangle_within_field else '_'
            # if x_bin_number % 4 == 0:  # limit printing of tiles in console
            print(ascii_char_to_draw, end='')

            if not is_rectangle_within_field:
                continue

            tile.save(damage_img=damage_area.mask_img,
                      field_img=tif_wrapper.img,
                      tile_output_dir=config.BASE_OUTPUT_DATA_DIR_PATH)

    print(f'Tiles saved!')



if __name__ == '__main__':
    main()
    cv2.waitKey()
    print('Done! Bye!')


