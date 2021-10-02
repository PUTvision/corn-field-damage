import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from geotiff import GeoTiff
import geopandas


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
        self.bounding_box = CoordinatesRect.from_2_tuples(self.geo_tiff.tif_bBox)

        self.img = cv2.imread(tif_file_path)
        self.img_size_y_pixels, self.img_size_x_pixels, _ = self.img.shape

        self._pixels_per_epsg_x = 1 / self.spacial_resolution_x_in_meters  # so the same as pixels for one meter in x
        self._pixels_per_epsg_y = 1 / self.spacial_resolution_y_in_meters  # so the same as pixels for one meter in y

    def print_stats(self):
        print(f'tif EPSG32633 bounding box = {self.bounding_box}')
        print(f'TIF image size xy in pixels: {self.img_size_y_pixels, self.img_size_x_pixels}')
        print(f'TIF image size xy in meters: '
              f'{self.bounding_box.get_x_distance_in_meters():.3f}, '
              f'{self.bounding_box.get_y_distance_in_meters():.3f}')
        print(f'TIF image resolution xy [meters per pixel]: '
              f'{self.spacial_resolution_x_in_meters:.3f}, {self.spacial_resolution_y_in_meters:.3f}')

    @property
    def spacial_resolution_x_in_meters(self) -> float:
        return self.bounding_box.get_x_distance_in_meters() / (self.img_size_x_pixels - 1)

    @property
    def spacial_resolution_y_in_meters(self) -> float:
        return self.bounding_box.get_y_distance_in_meters() / (self.img_size_y_pixels - 1)

    @property
    def get_square_meters_per_pixel(self) -> float:
        pixel_area_square_meters = self.spacial_resolution_x_in_meters * self.spacial_resolution_y_in_meters
        return pixel_area_square_meters

    def transform_polygon_to_xy_pixels(self, polygon: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Transform coordinates polygon to pixels contour (with cv2 format)
        :param polygon: tuple with two lists (x and y points respoectively)
        :return: 2D contour
        """
        x_epsg, y_epsg = polygon
        yx_px = np.zeros((len(x_epsg), 2), dtype=int)
        x_left = self.bounding_box.x_left
        y_upper = self.bounding_box.y_upper
        for i in range(len(x_epsg)):
            yx_px[i][0] = round((x_epsg[i] - x_left) * self._pixels_per_epsg_x)
            yx_px[i][1] = round((y_upper - y_epsg[i]) * self._pixels_per_epsg_y)

        return yx_px


class FieldArea:
    """ Wraps gpkg file with entire filed area mask (obszar.gpkg) """
    def __init__(self, file_path):
        self.data = geopandas.read_file(file_path)
        self.mask_img = None

    def get_polygon_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        area_polygon = self.data.to_numpy()[0][0][0]
        x_epsg, y_epsg = area_polygon.exterior.coords.xy
        return x_epsg, y_epsg

    def create_mask_for_tif(self, tif_wrapper: GeoTiffImageWrapper, erode_contour_size=30, show=False):
        shape = tif_wrapper.img.shape[:2]
        self.mask_img = np.zeros(shape, np.uint8)
        polygon = self.get_polygon_xy()
        yx_px = tif_wrapper.transform_polygon_to_xy_pixels(polygon)
        cv2.fillPoly(self.mask_img, pts=[yx_px], color=255)

        if erode_contour_size:
            # erosion because field area goes over the real border of the field sometime
            self.mask_img = cv2.erode(self.mask_img, np.ones((erode_contour_size, erode_contour_size), np.uint8))

        if show:
            small_height = 512
            area_mask_small = cv2.resize(self.mask_img, (small_height, int(512 / shape[1] * shape[0])))
            cv2.imshow('area_mask_small', area_mask_small)
            cv2.waitKey()


class DamageArea:
    """ Wraps gpkg file with damage filed area mask (szkody_placowe.gpkg) """
    def __init__(self, file_path):
        self.data = geopandas.read_file(file_path)


def main():
    BASE_DATA_DIR_PATH = "/home/przemek/Downloads/corn2/kukurydza_5_ha/"
    tif_file_path = os.path.join(BASE_DATA_DIR_PATH, 'fotomapa.tif')
    field_area_file_path = os.path.join(BASE_DATA_DIR_PATH, 'obszar.gpkg')
    damage_area_file_path = os.path.join(BASE_DATA_DIR_PATH, 'szkody_placowe.gpkg')

    tif_wrapper = GeoTiffImageWrapper(tif_file_path=tif_file_path)
    tif_wrapper.print_stats()

    field_area = FieldArea(field_area_file_path)
    damage_area = DamageArea(damage_area_file_path)

    field_area.create_mask_for_tif(tif_wrapper, show=True)


if __name__ == '__main__':
    main()
    print('end')


