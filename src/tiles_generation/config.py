import enum
import platform


hostname = platform.node()
if hostname == 'przemek-PC':
    # BASE_DATA_DIR_PATH = "/media/data/nextcloud/Wspoldzielone/PP/corn/raw"  # directory with data as on google drive
    BASE_DATA_DIR_PATH = "/media/data/nextcloud/Wspoldzielone/PP/corn/raw"  # directory with data as on google drive
    # BASE_OUTPUT_DATA_DIR_PATH = "/media/data/local/corn/processed_stride1152_v2"  # base directory where to save the processed data
    BASE_OUTPUT_DATA_DIR_PATH = "/tmp/testcorn"  # base directory where to save the processed data
    # MODEL_PATH = '/media/data/local/corn/processed_stride768/model_cpu__trained_without_10ha'
    MODEL_PATH = '/media/data/local/corn/processed_stride768_v2/model_cpu_deeplabv3'
elif hostname == 'przemek-notebook':
    BASE_DATA_DIR_PATH = "/media/przemek/data/corn_data/raw"  # directory with data as on google drive
    BASE_OUTPUT_DATA_DIR_PATH = "/media/przemek/data/corn_data/processed"  # base directory where to save the processed data
    # MODEL_PATH = '/home/przemek/Desktop/model_cpu'
else:
    raise Exception("Error! Please specify BASE_DATA_DIR_PATH and BASE_OUTPUT_DATA_DIR_PATH in config.py!")


TIF_FILE_NAME = 'fotomapa.tif'
FIELD_AREA_FILE_NAME = 'obszar.gpkg'
DAMAGE_AREA_FILE_NAME = 'szkody_placowe.gpkg'
POINT_DAMAGE_AREA_FILE_NAME = 'szkody_punktowe.gpkg'
DAMAGE_AREA_FROM_NN_FILE_NAME = 'szkody_placowe_nn.gpkg'

SUBDIRECTORIES_TO_PROCESS = [
    # 'Kukurydza_RGB_9_ha',
    # 'Kukurydza_RGB_134_ha',

    "kukurydza_5_ha",
    # "kukurydza_10_ha",
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


class TileSizeType(enum.Enum):
    SMALL = enum.auto()
    NORMAL = enum.auto()
    BIG = enum.auto()


def get_tile_size_for_type(tile_size_type: TileSizeType) -> int:
    return {
        TileSizeType.SMALL: (512 + 256) // 2,
        TileSizeType.NORMAL: (512 + 256),
        TileSizeType.BIG: (512 + 256) // 2 * 3,
    }[tile_size_type]


TILE_SIZE = get_tile_size_for_type(TileSizeType.NORMAL)  # in pixels
TILE_STRIDE = TILE_SIZE  # in pixels. Same as tile size, so there is no overlap in the data


MINIMUM_TILE_FIELD_COVERAGE_PERCENTAGE = 50  # 0-100. Minimal percentage of corn field in the tile to save it

COLOR_VALUE__DAMAGED_AREA_ON_TILE_MASK = 255  # color of damaged area on the image with mask for the tile
COLOR_VALUE__NOT_FIELD_AREA_ON_TILE_MASK = 127  # color of pixels outside of field on the image with mask for the tile (so pixels that are neither good nor damaged corn)
