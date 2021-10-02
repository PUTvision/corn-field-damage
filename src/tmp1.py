import os

import cv2
import numpy as np
from geotiff import GeoTiff
import geopandas as gp
import geopandas
from pyproj import Transformer



base_data_dir_path = "/home/przemek/Downloads/corn2/kukurydza_5_ha/"
tif_file_path = os.path.join(base_data_dir_path, 'fotomapa.tif')

geo_tiff = GeoTiff(tif_file_path)
print(f'fotomapa_size = {geo_tiff.tif_bBox}')
(x_left, y_upper), (x_right, y_bottom) = geo_tiff.tif_bBox

x_diff_meters = x_right - x_left
y_diff_meters = y_upper - y_bottom



img = cv2.imread(tif_file_path)
dir(img)
print(img.shape)
# subimg = img[1000:1100, 1000:1100]
# cv2.imshow('subimg', subimg)
# cv2.waitKey()

y_pixels, x_pixels, _ = img.shape
spacial_resolution_xy_meters = (x_diff_meters / (x_pixels - 1)), (y_diff_meters / (y_pixels - 1))
pixel_area_square_meters = spacial_resolution_xy_meters[0] * spacial_resolution_xy_meters[1]


area_file_path = os.path.join(base_data_dir_path, 'obszar.gpkg')
gp_area = geopandas.read_file(area_file_path)

damage_file_path = os.path.join(base_data_dir_path, 'szkody_placowe.gpkg')
gp_damage = geopandas.read_file(damage_file_path)


area_polygon = gp_area.to_numpy()[0][0][0]
x_epsg, y_epsg = area_polygon.exterior.coords.xy

pixels_per_x_epsg = 1 / spacial_resolution_xy_meters[0]
pixels_per_y_epsg = 1 / spacial_resolution_xy_meters[1]

yx_px = np.zeros((len(x_epsg), 2))
for i in range(len(x_epsg)):
    yx_px[i][0] = (x_epsg[i] - x_left) * pixels_per_x_epsg
    yx_px[i][1] = (y_upper - y_epsg[i]) * pixels_per_y_epsg

yx_px = yx_px.astype(int)


area_mask = np.zeros(img.shape[:2], np.uint8)
# cv2.fillPoly(area_mask, pts=[yx_px], color=55)

# area_mask_small = cv2.resize(area_mask, (y_pixels//20, x_pixels//20))
# cv2.imshow('area_mask_small', area_mask_small)
# cv2.waitKey()

cv2.polylines(img, pts=[yx_px], isClosed=True, color=55)
cv2.imwrite("/tmp/img1.tiff", img)





print('end')
print('end2')


