import cv2
import segmentation_models_pytorch as smp
import torch
import numpy as np

from model_training_v2.common import corn_dataset
from model_training_v2.common import model_definition
from tiles_generation.common import util


class FieldDamageSegmentationModel:
    NUMBER_OF_SEGMENTATION_CLASSES = corn_dataset.NUMBER_OF_SEGMENTATION_CLASSES

    def __init__(self, model_file_path, model_type):
        self.model, _ = model_definition.get_model_with_params(model_type=model_type)
        self.model.load_state_dict(torch.load(model_file_path))
        self.model.eval()

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
