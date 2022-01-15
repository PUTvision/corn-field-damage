import math
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import cv2
from geotiff import GeoTiff
from skimage import io
import skimage

from tiles_generation.common import config
from tiles_generation.common import util


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
    def __init__(self, file_path: str, number_of_channels: int = 3):
        self.number_of_channels = number_of_channels
        self.geo_tiff = GeoTiff(file_path)
        assert self.geo_tiff.crs_code == 32633  # Processing supports only EPSG32633
        self.coord_bounding_box = CoordinatesRect.from_2_tuples(self.geo_tiff.tif_bBox)

        assert os.path.isfile(file_path)

        self.img = self.load_and_scale_img(file_path)

        self.img_size_y_pixels, self.img_size_x_pixels, _ = self.img.shape

        self._pixels_per_epsg_x = 1 / self.spacial_resolution_x_in_meters  # so the same as pixels for one meter in x
        self._pixels_per_epsg_y = 1 / self.spacial_resolution_y_in_meters  # so the same as pixels for one meter in y
        self.print_stats()

    def load_and_scale_img(self, file_path):
        # self.img = cv2.imread(file_path)  # opencv fails to load image larger than a few GB
        # assert self.img.data
        if self.number_of_channels > 1:
            img = skimage.io.imread(file_path)[:, :, :self.number_of_channels].copy()
        else:
            img = skimage.io.imread(file_path)[..., np.newaxis].copy()

        # assume x and y resolutions are equal
        x_resolution = self.coord_bounding_box.get_x_distance_in_meters() / (img.shape[1] - 1)
        print(f'TIF image resolution x before scaling: {x_resolution:.3f} [meters per pixel]')
        expected_resolution = config.FINAL_SCALED_IMAGE_RESOLUTION__METER_PER_PIXEL
        scaling_factor = x_resolution / expected_resolution
        new_shape = int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)
        img_scaled = cv2.resize(img, dsize=new_shape, interpolation=cv2.INTER_CUBIC)

        if self.number_of_channels == 1:
            img_scaled = img_scaled[..., np.newaxis]

        return img_scaled

    def print_stats(self):
        print(f'TIF EPSG32633 bounding box = {self.coord_bounding_box}')
        print(f'TIF image size xy in pixels: {self.img_size_y_pixels, self.img_size_x_pixels}')
        print(f'TIF image size xy in meters: '
              f'{self.coord_bounding_box.get_x_distance_in_meters():.3f}, '
              f'{self.coord_bounding_box.get_y_distance_in_meters():.3f}')
        print(f'TIF image resolution xy [meters per pixel]: '
              f'{self.spacial_resolution_x_in_meters:.3f}, {self.spacial_resolution_y_in_meters:.3f}')

    def convert_distance_in_meters_to_pixels(self, distance) -> float:
        # assuming x and y resolutions are close enough
        return distance / self.spacial_resolution_x_in_meters

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

    def transform_polygons_epsg_to_yx_pixels(self, polygons: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
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

    def transform_contours_yx_pixels_to_epsg(self, polygons):
        x_left = self.coord_bounding_box.x_left
        y_upper = self.coord_bounding_box.y_upper

        polygons_epsg = []
        for polygon_3d in polygons:
            polygon = polygon_3d.squeeze()
            polygon_epsg = []
            for i in range(len(polygon)):
                yx_px = polygon[i]
                x_epsg = yx_px[0] / self._pixels_per_epsg_x + x_left
                y_epsg = -(yx_px[1] / self._pixels_per_epsg_y - y_upper)
                polygon_epsg.append([x_epsg, y_epsg])
            polygons_epsg.append(polygon_epsg)
        return polygons_epsg

    def apply_mask(self, mask_img, show=False):
        # rgb files with 3 channels are uint8
        self.img = cv2.bitwise_and(self.img, self.img, mask=mask_img)
        util.show_small_img(img=self.img, name='field_masked', show=show)


class GeoTiffImageWrapperNDVI(GeoTiffImageWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_mask(self, mask_img, show=False):
        # one channel NDVI file is float32
        self.img = cv2.copyTo(self.img, mask_img)

        # set pixels outside of the real corn field to some other value
        # we need to split this operation into a few steps, because np.where consumes too much RAM!
        STEP = 5000
        for i in range(math.ceil(self.img.shape[0] / STEP)):
            start_index = i * STEP
            end_index = (i+1) * STEP
            self.img[start_index:end_index, :][np.where(mask_img[start_index:end_index, :] == 0)] = \
                config.COLOR_VALUE__NOT_FIELD_AREA_ON_NDVI_IMAGE

        util.show_small_img(img=self.img, name='field_masked', show=show)

    def transform_to_rgb_tif_wrapper_pixels_space(self, tif_wrapper: GeoTiffImageWrapper):
        cb = self.coord_bounding_box
        mapped_corners = tif_wrapper.transform_polygons_epsg_to_yx_pixels(
            [[
                [cb.x_left,  cb.x_right,  cb.x_right],
                [cb.y_upper, cb.y_upper,  cb.y_bottom],
            ]])

        old_shape = self.img.shape[:2]
        max_old_x = old_shape[1] - 1
        max_old_y = old_shape[0] - 1

        new_shape = tif_wrapper.img.shape[:2]

        mc = mapped_corners[0]
        pts1 = np.float32([ [0,        0       ], [max_old_x, 0],       [max_old_x, max_old_y] ])
        pts2 = np.float32([ [mc[0][0], mc[0][1]], [mc[1][0], mc[1][1]], [mc[2][0], mc[2][1]]   ])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(self.img, M, new_shape[::-1])
        self.img = dst



