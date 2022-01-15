import platform
import os
from tiles_generation.common import config

from model_training_v2.common import model_definition


CONFIG_FILE_PATH = os.path.abspath(__file__)


BASE_DATA_DIR_PATH = None
MODEL_PATH = None

hostname = platform.node()
if hostname == 'przemek-PC':
    BASE_DATA_DIR_PATH = "/media/data/nextcloud/Wspoldzielone/PP/corn/raw"  # directory with data as on cloud
    MODEL_PATH = '/media/data/local/corn/out/from_drive/model_UNET_PLUS_PLUS__EFFICIENT_NET_B3__big_lr_as_always/model.zip'
    MODEL_TYPE = model_definition.ModelType.UNET_PLUS_PLUS__EFFICIENT_NET_B3


if not (BASE_DATA_DIR_PATH or MODEL_PATH):
    raise Exception(f"Error! Please specify BASE_DATA_DIR_PATH and BASE_OUTPUT_DATA_DIR_PATH in config.py!\n"
                    f"({os.path.abspath(__file__)})")


TIF_FILE_NAME = 'fotomapa.tif'
FIELD_AREA_FILE_NAME = 'obszar.gpkg'
DAMAGE_AREA_FROM_NN_FILE_NAME = 'szkody_placowe_nn.gpkg'


SUBDIRECTORIES_TO_PROCESS = [

    #
    # "kukurydza_5_ha",
    # "kukurydza_10_ha",
    # "Kukurydza_RGB_9_ha",
    "kukurydza_11_ha",
    "kukurydza_13_ha",
    "kukurydza_15_ha",
    "kukurydza_18_ha",
    "kukurydza_25_ha",
    "kukurydza_38_ha",
    "kukurydza_60_ha",

    "Kukurydza_RGB_25_ha",
    "Kukurydza_RGB_25.5_ha",
    'Kukurydza_RGB_66_ha',

    # only for NVDI training (not for normal training)
    # 'Kukurydza_RGB_NDVI_50_ha',
]

TILE_SIZE = 512


# a little hacky, but we are using some files from tile_generation module, therefore we need to overwrite config
config.TILE_SIZE = TILE_SIZE
config.FINAL_SCALED_IMAGE_RESOLUTION__METER_PER_PIXEL = 0.03
