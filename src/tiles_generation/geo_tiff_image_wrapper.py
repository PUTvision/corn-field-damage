import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import cv2
from geotiff import GeoTiff
from skimage.io import imread

import util


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
    def __init__(self, file_path: str):
        self.geo_tiff = GeoTiff(file_path)
        assert self.geo_tiff.crs_code == 32633  # Processing supports only EPSG32633
        self.coord_bounding_box = CoordinatesRect.from_2_tuples(self.geo_tiff.tif_bBox)

        assert os.path.isfile(file_path)

        # self.img = cv2.imread(file_path)  # opencv fails to load image larger than a few GB
        # assert self.img.data
        self.img = imread(file_path)[:, :, :3].copy()

        self.img_size_y_pixels, self.img_size_x_pixels, _ = self.img.shape

        self._pixels_per_epsg_x = 1 / self.spacial_resolution_x_in_meters  # so the same as pixels for one meter in x
        self._pixels_per_epsg_y = 1 / self.spacial_resolution_y_in_meters  # so the same as pixels for one meter in y
        self.print_stats()

    def print_stats(self):
        print(f'TIF EPSG32633 bounding box = {self.coord_bounding_box}')
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

    def transform_polygons_to_xy_pixels(self, polygons: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
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
