import os

import numpy as np
import cv2

import config


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

        cv2.imwrite(img=field_roi, filename=tile_img_file_path)
        cv2.imwrite(img=damage_roi, filename=tile_mask_file_path)
