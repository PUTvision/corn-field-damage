import enum
import os.path
import sys
from dataclasses import dataclass
from functools import partial
from typing import Optional
from torch import nn, optim

import segmentation_models_pytorch as smp

from model_training_v2.common.corn_dataset import NUMBER_OF_SEGMENTATION_CLASSES


class ModelType(enum.Enum):
    UNET = enum.auto()
    UNET_PLUS_PLUS = enum.auto()  # efficientnet-b0
    UNET_PLUS_PLUS__EFFICIENT_NET_B0 = enum.auto()
    UNET_PLUS_PLUS__EFFICIENT_NET_B1 = enum.auto()
    UNET_PLUS_PLUS__EFFICIENT_NET_B2 = enum.auto()
    UNET_PLUS_PLUS__EFFICIENT_NET_B3 = enum.auto()
    UNET_PLUS_PLUS__EFFICIENT_NET_B4 = enum.auto()
    UNET_PLUS_PLUS__RESNET18 = enum.auto()
    UNET_PLUS_PLUS__RESNET50 = enum.auto()
    UNET_PLUS_PLUS__DENSENET121 = enum.auto()
    UNET_PLUS_PLUS__DENSENET201 = enum.auto()
    DEEP_LAB_V3 = enum.auto()
    PAN = enum.auto()
    DEEP_LAB_V3_PLUS = enum.auto()
    LINKNET = enum.auto()
    FPN = enum.auto()
    SEGFORMER_B0 = enum.auto()
    SEGFORMER_B3 = enum.auto()


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


def get_model_with_params(model_type: ModelType, in_channels=3, tile_size=None) -> tuple:
    params = ModelParams(model_type=model_type)

    if model_type == ModelType.UNET:
        params.batch_size = 3
        model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS:
        params.batch_size = 3
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS__EFFICIENT_NET_B0:
        params.batch_size = 3
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS__EFFICIENT_NET_B1:
        params.batch_size = 3
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b1",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS__EFFICIENT_NET_B2:
        params.batch_size = 3
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS__EFFICIENT_NET_B3:
        params.batch_size = 3
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS__EFFICIENT_NET_B4:
        params.batch_size = 3
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS__RESNET18:
        params.batch_size = 1
        model = smp.UnetPlusPlus(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='ssl',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS__RESNET50:
        params.batch_size = 1
        model = smp.UnetPlusPlus(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='ssl',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS__DENSENET121:
        params.batch_size = 1
        model = smp.UnetPlusPlus(
            encoder_name="densenet121",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS__DENSENET201:
        params.batch_size = 1
        model = smp.UnetPlusPlus(
            encoder_name="densenet201",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS:
        params.batch_size = 3
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.UNET_PLUS_PLUS:
        params.batch_size = 3
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.DEEP_LAB_V3:
        params.batch_size = 2
        model = smp.DeepLabV3(
            encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.PAN:
        params.batch_size = 2
        model = smp.PAN(
            encoder_name="timm-gernet_s",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.DEEP_LAB_V3_PLUS:
        params.batch_size = 2
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.LINKNET:
        params.batch_size = 2
        model = smp.Linknet(
            encoder_name="efficientnet-b0",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type == ModelType.FPN:
        params.batch_size = 2
        model = smp.MAnet(
            encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
    elif model_type in [ModelType.SEGFORMER_B3, ModelType.SEGFORMER_B0]:
        # As segformer is in a separate library within submodule, we need to add it to path manually.
        # It is added here to allow working with other modules without need for this module
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        segformer_submodule_path = os.path.join(SCRIPT_DIR, '..', 'segmentation_pytorch')
        sys.path.insert(0, segformer_submodule_path)

        from model_training_v2.segmentation_pytorch.configs.segformer_config import config as segformer_cfg
        from model_training_v2.segmentation_pytorch.models.segformer import Segformer

        if tile_size is None:
            raise Exception("For Segformer model tile size cannot be None!")

        segformer_cfg.DATASET.NUM_CLASSES = NUMBER_OF_SEGMENTATION_CLASSES
        segformer_cfg.DATASET.CROP_SIZE = (tile_size, tile_size)

        if model_type == ModelType.SEGFORMER_B3:
            SEG_CFG = segformer_cfg.MODEL.B3
        elif model_type == ModelType.SEGFORMER_B0:
            SEG_CFG = segformer_cfg.MODEL.B0
        else:
            raise Exception("Unknwon segformer type!")

        model = Segformer(
            num_classes=3,
            img_size=tile_size,
            # pretrained = SEG_CFG.PRETRAINED,
            pretrained = None,
            patch_size = segformer_cfg.MODEL.PATCH_SIZE,
            embed_dims = SEG_CFG.CHANNEL_DIMS,
            num_heads = SEG_CFG.NUM_HEADS,
            mlp_ratios = SEG_CFG.MLP_RATIOS,
            qkv_bias = SEG_CFG.QKV_BIAS,
            depths = SEG_CFG.DEPTHS,
            sr_ratios = SEG_CFG.SR_RATIOS,
            drop_rate = SEG_CFG.DROP_RATE,
            drop_path_rate = SEG_CFG.DROP_PATH_RATE,
            decoder_dim = SEG_CFG.DECODER_DIM,
            norm_layer = partial(nn.LayerNorm, eps=1e-6),
        )
        params.mask_scalling_factor = 4.0
        params.metrics_activation = 'softmax2d'
        params.loss_fnc = smp.utils.losses.CrossEntropyLoss  # without soft2d_out and with activation
        params.batch_size = 1


        # SOME old segformer - not working?
        # from mmseg.models import build_segmentor

        # norm_cfg = dict(type='SyncBN', requires_grad=True)

        # cfg_model = dict(
        #     type='EncoderDecoder',
        #     pretrained='/home/przemek/Projects/pp/corn-field-damage/tmp2/SegFormer/pretrained/mit_b1.pth',
        #     backbone=dict(
        #         type='mit_b1',
        #         style='pytorch'),
        #     decode_head=dict(
        #         type='SegFormerHead',
        #         in_channels=[64, 128, 320, 512],
        #         in_index=[0, 1, 2, 3],
        #         feature_strides=[4, 8, 16, 32],
        #         channels=128,
        #         dropout_ratio=0.1,
        #         # num_classes=150,  ## PA: changed
        #         num_classes=3,
        #         norm_cfg=norm_cfg,
        #         align_corners=False,
        #         decoder_params=dict(embed_dim=256),
        #         loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        # )

        # model = build_segmentor(
        #     cfg_model,
        #     train_cfg=None,
        #     test_cfg=None)

    else:
        raise Exception(f"Unknown model type: {model_type.name}")

    return model, params


def main():
    model, model_params = get_model_with_params(ModelType.SEGFORMER_B3, in_channels=3, tile_size=512)

    from model_training_v2.common import model_training
    from model_training_v2.common.model_training_results import ModelTrainingResults

    model_trainer = model_training.ModelTrainer(
        model=model,
        device='cpu',
        model_params=model_params,
        res=ModelTrainingResults(),
        )


if __name__ == '__main__':
    main()

