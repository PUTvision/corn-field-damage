import platform

hostname = platform.node()
if hostname == 'przemek-PC':
    BASE_DATA_DIR_PATH = "/media/data/nextcloud/Wspoldzielone/PP/corn/raw"  # directory with data as on google drive
    BASE_OUTPUT_DATA_DIR_PATH = "/media/data/local/corn/processed"  # base directory where to save the processed data
    MODEL_PATH = '/home/przemek/Desktop/model_cpu'
elif hostname == 'przemek-notebook':
    BASE_DATA_DIR_PATH = "/media/przemek/data/corn_data/raw"  # directory with data as on google drive
    BASE_OUTPUT_DATA_DIR_PATH = "/media/przemek/data/corn_data/processed"  # base directory where to save the processed data
    MODEL_PATH = '/home/przemek/Desktop/model_cpu'
else:
    raise Exception("Error! Please specify BASE_DATA_DIR_PATH and BASE_OUTPUT_DATA_DIR_PATH in config.py!")


TIF_FILE_NAME = 'fotomapa.tif'
FIELD_AREA_FILE_NAME = 'obszar.gpkg'
DAMAGE_AREA_FILE_NAME = 'szkody_placowe.gpkg'
POINT_DAMAGE_AREA_FILE_NAME = 'szkody_punktowe.gpkg'
DAMAGE_AREA_FROM_NN_FILE_NAME = 'szkody_placowe_nn.gpkg'

SUBDIRECTORIES_TO_PROCESS = [
    # "kukurydza_5_ha",
    "kukurydza_10_ha",
    # "kukurydza_11_ha",
    # "kukurydza_13_ha",
    # "kukurydza_15_ha",
    # "kukurydza_18_ha",
    # "kukurydza_25_ha",
    # "kukurydza_38_ha",
    # "kukurydza_60_ha",
]

POINT_DAMAGE_RADIUS_METERS = 1.5 / 2  # ~ 1.5m diameter on average - 50 pixels
FIELD_BORDER_EROSION_SIZE_PIXELS = 70  # [in pixels] Erode the field border, because sometimes it goes too far
FINAL_SCALED_IMAGE_RESOLUTION__METER_PER_PIXEL = 0.03  # how many meters for one pixel we want in our scaled image

TILE_STRIDE = 768  # in pixels
# TILE_STRIDE = 256  # in pixels
TILE_SIZE = (512 + 256)  # in pixels
MINIMUM_TILE_FIELD_COVERAGE_PERCENTAGE = 75  # 0-100. Minimal percentage of corn field in the tile to save it

COLOR_VALUE__DAMAGED_AREA_ON_TILE_MASK = 255  # color of damaged area on the image with mask for the tile
COLOR_VALUE__NOT_FIELD_AREA_ON_TILE_MASK = 127  # color of pixels outside of field on the image with mask for the tile (so pixels that are neither good nor damaged corn)
