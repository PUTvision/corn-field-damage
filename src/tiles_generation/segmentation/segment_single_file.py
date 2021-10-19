import os

import geopandas
import numpy as np
import cv2

from tiles_generation import util
from tiles_generation.segmentation.model import FieldDamageSegmentationModel
import config


def main():
    model = FieldDamageSegmentationModel(config.MODEL_PATH)
    tile_img_bgr = cv2.imread("/home/przemek/Desktop/20210824-pole6/raw/4.png")  # already rgb!
    tile_img_rgb = cv2.cvtColor(tile_img_bgr, cv2.COLOR_BGR2RGB)
    util.show_small_img(tile_img_rgb, name="tile_img_bgr", reverse_rgb_bgr=True)
    predicted_mask_binary = model.predict_damage(tile_img_rgb, show=True, debug=True)


if __name__ == '__main__':
    main()
    cv2.waitKey()
    print('Done! Bye!')


