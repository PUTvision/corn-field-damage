import enum
from dataclasses import dataclass
from typing import Optional

import segmentation_models_pytorch as smp

from corn_dataset import NUMBER_OF_SEGMENTATION_CLASSES


class ModelType(enum.Enum):
    UNET = enum.auto()
    UNET_PLUS_PLUS = enum.auto()
    DEEP_LAB_V3 = enum.auto()
    PAN = enum.auto()
    # MANET = enum.auto()
    # LINKNET = enum.auto()
    # FPN = enum.auto()
    # SEGFORMER = enum.auto()


def get_mask_scalling_for_model(model_type: ModelType):
    # if model_type == ModelType.SEGFORMER:
    #     return 4
    return None


@dataclass
class ModelParams:
    model_type: ModelType
    loss_fnc: type = smp.utils.losses.DiceLoss
    mask_scalling_factor: Optional[float] = None
    metrics_activation: Optional[str] = None
    batch_size: int = 1

    def get_model_file_name(self):
        return 'model_' + self.model_type.name


def get_model_with_params(model_type: ModelType) -> tuple:
    params = ModelParams(model_type = model_type)

    if model_type == ModelType.UNET:
        params.batch_size = 3
        model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS:
        params.batch_size = 3
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        #     aux_params={'pooling': 'max', 'classes': 3}
        #     aux_params={'dropout':0.1, 'classes':3}
        )
    elif model_type == ModelType.DEEP_LAB_V3:
        model = smp.DeepLabV3(
            encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.PAN:
        model = smp.PAN(
            encoder_name="timm-gernet_s",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )

    # elif SEGFORMER:
        # this one is good?
        # import sys
        # sys.path.insert(0,'/home/przemek/Projects/pp/corn-field-damage/src/model_training/segmentation_pytorch-main')

        # from configs.segformer_config import config as cfg
        # from models.segformer import Segformer

        # SEG_CFG = cfg.MODEL.B3
        # model = Segformer(
        #     num_classes = 3,
        #     img_size = 512,  #cfg.DATASET.CROP_SIZE[0],

        #     # pretrained = SEG_CFG.PRETRAINED,
        #     pretrained = None,
        #     patch_size = cfg.MODEL.PATCH_SIZE,
        #     embed_dims = SEG_CFG.CHANNEL_DIMS,
        #     num_heads = SEG_CFG.NUM_HEADS,
        #     mlp_ratios = SEG_CFG.MLP_RATIOS,
        #     qkv_bias = SEG_CFG.QKV_BIAS,
        #     depths = SEG_CFG.DEPTHS,
        #     sr_ratios = SEG_CFG.SR_RATIOS,
        #     drop_rate = SEG_CFG.DROP_RATE,
        #     drop_path_rate = SEG_CFG.DROP_PATH_RATE,
        #     decoder_dim = SEG_CFG.DECODER_DIM,
        #     norm_layer = partial(nn.LayerNorm, eps=1e-6),
        # ).to(DEVICE)
        # metrics_activation = 'softmax2d'

        #     params.mask_scalling_factor = 4.0
        #     params.metrics_activation = 'softmax2d'
        #     patams.loss_fnc = smp.utils.losses.CrossEntropyLoss()
    else:
        raise Exception(f"Unknown model type: {model_type.name}")

    return model, params
