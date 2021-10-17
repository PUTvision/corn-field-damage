import math
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
            for interior in area_polygon.interiors:
                interiors.append(interior.coords.xy)
            x_epsg, y_epsg = area_polygon.exterior.coords.xy
            polygons.append((x_epsg, y_epsg))
        return polygons, interiors

    def create_mask_for_tif(self, tif_wrapper: GeoTiffImageWrapper, erode_contour_size: int = 0, show=False):
        shape = tif_wrapper.img.shape[:2]
        self.mask_img = np.zeros(shape, np.uint8)
        polygons, interiors = self._get_polygons_xy()
        yx_pixel_contours = tif_wrapper.transform_polygons_epsg_to_yx_pixels(polygons)
        yx_pixel_interiors = tif_wrapper.transform_polygons_epsg_to_yx_pixels(interiors)
        cv2.fillPoly(self.mask_img, pts=yx_pixel_contours, color=self.mask_color)
        cv2.fillPoly(self.mask_img, pts=yx_pixel_interiors, color=0)

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

    def apply_mask_on_img(self, img):
        img = cv2.bitwise_and(src1=img, src2=img, mask=self.mask_img)
        # util.show_small_img(tif_wrapper.img, 'tif_wrapper.img ')
        return img

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

    @staticmethod
    def _get_point_pixel_coordinates(point_damage_file_path: str, tif_wrapper: GeoTiffImageWrapper):
        point_damage_data = geopandas.read_file(point_damage_file_path)
        points = point_damage_data.to_numpy()
        points_pixel_coordinates = []
        for point_list in points:
            point = point_list[0]
            xy = [point.xy[0][0]], [point.xy[1][0]]
            points_pixel_coordinates += tif_wrapper.transform_polygons_epsg_to_yx_pixels([xy])
        return points_pixel_coordinates

    def _create_mask_for_point_damages(self, point_damage_file_path: str, tif_wrapper: GeoTiffImageWrapper):
        points_pixel_coordinates = self._get_point_pixel_coordinates(
            point_damage_file_path=point_damage_file_path, tif_wrapper=tif_wrapper)

        radius_pixels = round(tif_wrapper.convert_distance_in_meters_to_pixels(config.POINT_DAMAGE_RADIUS_METERS))
        for point in points_pixel_coordinates:
            yx = point[0]
            cv2.circle(self.mask_img, center=yx, radius=radius_pixels, thickness=-1, color=self.mask_color)

    def create_mask_for_tif_and_area(self,
                                     tif_wrapper: GeoTiffImageWrapper,
                                     field_area: FieldArea,
                                     point_damage_file_path: str,
                                     show=False):
        super().create_mask_for_tif(tif_wrapper=tif_wrapper, show=show)

        self._create_mask_for_point_damages(
            point_damage_file_path=point_damage_file_path, tif_wrapper=tif_wrapper)

        damaged_area_m2 = self.calculate_damaged_pixels_count(self.mask_img) * tif_wrapper.get_square_meters_per_pixel
        total_field_size = cv2.countNonZero(field_area.mask_img) * tif_wrapper.get_square_meters_per_pixel
        print(f'Damaged area = {damaged_area_m2:.3f} m^2')
        print(f'Total field area = {total_field_size:.3f} m^2')
        print(f'Total damaged as percentage: {damaged_area_m2/total_field_size * 100:.3f} %')

        # set pixels outside of the real corn field to some other value
        # we need to split this operation into a few steps, because np.where consumes too much RAM!
        STEP = 5000
        for i in range(math.ceil(field_area.mask_img.shape[0] / STEP)):
            start_index = i * STEP
            end_index = (i+1) * STEP
            self.mask_img[start_index:end_index, :][np.where(field_area.mask_img[start_index:end_index, :] == 0)] = \
                config.COLOR_VALUE__NOT_FIELD_AREA_ON_TILE_MASK

        util.show_small_img(self.mask_img, name='damage', show=show)

    @classmethod
    def calculate_damaged_pixels_count(cls, mask_img):
        if mask_img is None:
            raise Exception("Mask not create yet!")
        # unique, counts = np.unique(self.mask_img, return_counts=True)
        damaged_pixels = np.where(mask_img == config.COLOR_VALUE__DAMAGED_AREA_ON_TILE_MASK)
        damaged_pixels_count = damaged_pixels[0].shape[0]
        return damaged_pixels_count
