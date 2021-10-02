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

    small_height = 512
    shape = img.shape[:2]
    img_small = cv2.resize(img, (small_height, int(512 / shape[1] * shape[0])))
    cv2.imshow(name, img_small)
    cv2.waitKey()

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
            x_left = self.bounding_box.x_left
            y_upper = self.bounding_box.y_upper
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

        util.show_small_img(img=self.mask_img, name='area_mask_small', show=show)


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



def main():
    BASE_DATA_DIR_PATH = "/home/przemek/Downloads/corn2/kukurydza_5_ha/"
    tif_file_path = os.path.join(BASE_DATA_DIR_PATH, 'fotomapa.tif')
    field_area_file_path = os.path.join(BASE_DATA_DIR_PATH, 'obszar.gpkg')
    damage_area_file_path = os.path.join(BASE_DATA_DIR_PATH, 'szkody_placowe.gpkg')

    tif_wrapper = GeoTiffImageWrapper(tif_file_path=tif_file_path)
    tif_wrapper.print_stats()

    field_area = FieldArea(field_area_file_path)
    damage_area = DamageArea(damage_area_file_path)

    damage_area.create_mask_for_tif(tif_wrapper, show=False)
    field_area.create_mask_for_tif(tif_wrapper, show=False)

    tif_wrapper.apply_mask(field_area.mask_img)

    


if __name__ == '__main__':
    main()
    cv2.waitKey()
    print('end')


