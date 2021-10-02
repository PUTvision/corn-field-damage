import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import geopandas
import numpy as np
import cv2

import config
import util
from geo_tiff_image_wrapper import GeoTiffImageWrapper
from tile import Tile


class BaseArea:
    def __init__(self, file_path, mask_color=255):
        self.data = geopandas.read_file(file_path)
        self.mask_img = None
        self.mask_color = mask_color

    def _get_polygons_xy(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        polygons = []
        interiors = []
        for shapely_polygon in self.data.to_numpy():
            area_polygon = shapely_polygon[0][0]
            x_epsg, y_epsg = area_polygon.exterior.coords.xy
            polygons.append((x_epsg, y_epsg))
        return polygons

    def create_mask_for_tif(self, tif_wrapper: GeoTiffImageWrapper, erode_contour_size: int = 0, show=False):
        shape = tif_wrapper.img.shape[:2]
        self.mask_img = np.zeros(shape, np.uint8)
        polygons = self._get_polygons_xy()
        yx_pixel_contours = tif_wrapper.transform_polygons_to_xy_pixels(polygons)
        cv2.fillPoly(self.mask_img, pts=yx_pixel_contours, color=self.mask_color)

        if erode_contour_size:
            # erosion because field area goes over the real border of the field sometime
            self.mask_img = cv2.erode(self.mask_img, np.ones((erode_contour_size, erode_contour_size), np.uint8))

        util.show_small_img(img=self.mask_img, name='area_small', show=show)


class FieldArea(BaseArea):
    """ Wraps gpkg file with entire filed area mask (obszar.gpkg) """
    def __init__(self, file_path):
        super().__init__(file_path)

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
        super().__init__(file_path, mask_color=config.COLOR_VALUE__DAMAGED_AREA_ON_TILE_MASK)

    def create_mask_for_tif_and_area(self, tif_wrapper: GeoTiffImageWrapper, field_area: FieldArea, show=False):
        super().create_mask_for_tif(tif_wrapper=tif_wrapper, show=show)

        damaged_area_m2 = self.calculate_damaged_pixels_count() * tif_wrapper.get_square_meters_per_pixel
        total_field_size = cv2.countNonZero(field_area.mask_img) * tif_wrapper.get_square_meters_per_pixel
        print(f'Damaged area = {damaged_area_m2:.3f} m^2')
        print(f'Total field area = {total_field_size:.3f} m^2')
        print(f'Damaged area as percentage: {damaged_area_m2/total_field_size * 100:.3f} %')

        # set pixels outside of the real corn field to some other value
        # we need to split this operation into a few steps, because np.where consumes too much RAM!
        STEP = 5000
        for i in range(math.ceil(field_area.mask_img.shape[0] / STEP)):
            start_index = i * STEP
            end_index = (i+1) * STEP
            self.mask_img[start_index:end_index, :][np.where(field_area.mask_img[start_index:end_index, :] == 0)] = \
                config.COLOR_VALUE__NOT_FIELD_AREA_ON_TILE_MASK

        util.show_small_img(self.mask_img, name='damage', show=show)

    def calculate_damaged_pixels_count(self):
        if self.mask_img is None:
            raise Exception("Mask not create yet!")
        # unique, counts = np.unique(self.mask_img, return_counts=True)
        damaged_pixels = np.where(self.mask_img == config.COLOR_VALUE__DAMAGED_AREA_ON_TILE_MASK)
        damaged_pixels_count = damaged_pixels[0].shape[0]
        return damaged_pixels_count
