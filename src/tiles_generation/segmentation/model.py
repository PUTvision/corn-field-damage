import cv2
import segmentation_models_pytorch as smp
import torch
import numpy as np

import util


class FieldDamageSegmentationModel:
    NUMBER_OF_SEGMENTATION_CLASSES = 3

    def __init__(self, model_file_path):
        # copied from jupyter notebook for training
        self.model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.NUMBER_OF_SEGMENTATION_CLASSES,  # model output channels (number of classes in your dataset)
            activation='softmax2d',  # ?
        )
        self.model.load_state_dict(torch.load(model_file_path))
        self.model.eval()
        # print(self.model)

    def predict_damage(self, img, show=False, debug=False):
        util.show_small_img(img, 'img', show=show)
        img = img.astype('float32')
        img /= 255
        img = img.transpose(2, 0, 1)

        img_batch = torch.tensor(img)[np.newaxis, :, :]
        with torch.no_grad():
            #     model_output = model(img_batch.to(DEVICE))
            model_output = self.model(img_batch)

        predicted_mask_damage = model_output[0][1].numpy() * 255
        predicted_mask_damage = predicted_mask_damage.astype(np.uint8)

        if debug:
            predicted_mask_healthy = model_output[0][0].numpy() * 255
            predicted_mask_healthy = predicted_mask_healthy.astype(np.uint8)
            util.show_small_img(predicted_mask_healthy, 'predicted_mask_healthy', show=show)

            predicted_mask_outside_field = model_output[0][2].numpy() * 255
            predicted_mask_outside_field = predicted_mask_outside_field.astype(np.uint8)
            util.show_small_img(predicted_mask_outside_field, 'predicted_mask_outside_field', show=show)

        util.show_small_img(predicted_mask_damage, 'predicted_mask_damage', show=show)
        _, predicted_mask_binary = cv2.threshold(predicted_mask_damage, thresh=255//2, maxval=255, type=cv2.THRESH_BINARY)
        util.show_small_img(predicted_mask_binary, 'predicted_mask_binary', show=show)
        return predicted_mask_binary
