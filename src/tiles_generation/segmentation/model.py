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

    def predict_damage(self, img, show=True):
        util.show_small_img(img, 'img', show=show)
        img = img.astype('float32')
        img /= 255
        img = img.transpose(2, 0, 1)

        img_batch = torch.tensor(img)[np.newaxis, :, :]
        with torch.no_grad():
            #     model_output = model(img_batch.to(DEVICE))
            model_output = self.model(img_batch)

        predicted_mask = model_output[0][1].numpy() * 255
        predicted_mask = predicted_mask.astype(np.uint8)
        util.show_small_img(predicted_mask, 'predicted_mask', show=show)
        _, predicted_mask_binary = cv2.threshold(predicted_mask, thresh=255//2, maxval=255, type=cv2.THRESH_BINARY)
        util.show_small_img(predicted_mask_binary, 'predicted_mask_binary', show=show)
        cv2.waitKey()
        return predicted_mask_binary
