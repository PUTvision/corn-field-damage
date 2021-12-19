import copy

import segmentation_models_pytorch as smp
import torch
from torch import nn
import numpy as np

from model_definition import ModelParams
from model_training_results import ModelTrainingResults


class ModelTrainer:
    def __init__(self, model, device, model_params: ModelParams, res: ModelTrainingResults):
        self.res = res
        self.loss = model_params.loss_fnc()
        self.model = model

        metrics_activation = model_params.metrics_activation
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5, name='IoU', activation=metrics_activation),
            smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[1, 2], name='IoU-0', activation=metrics_activation),
            smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0, 2], name='IoU-1', activation=metrics_activation),
            smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0, 1], name='IoU-2', activation=metrics_activation),
            # smp.utils.metrics.Fscore(threshold=0.5, activation=metrics_activation),  # dice_loss ~= 1 - Fscore
            smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[2], activation=metrics_activation),
            smp.utils.metrics.Accuracy(threshold=0.5, ignore_channels=[2], activation=metrics_activation),
            smp.utils.metrics.Recall(threshold=0.5, ignore_channels=[2], activation=metrics_activation),
            smp.utils.metrics.Precision(threshold=0.5, ignore_channels=[2], activation=metrics_activation),
        ]

        # optimizer = optim.SGD(model_fnn.parameters(), lr=0.0001, momentum=0.9)
        self.optimizer = torch.optim.Adam([
            dict(params=model.parameters())
        ])

        self.train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=device,
            verbose=True,
        )

        self.valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=self.loss,
            metrics=self.metrics,
            device=device,
            verbose=True,
        )

        self.test_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=self.loss,
            metrics=self.metrics,
            device=device,
            verbose=True,
        )

        for e in [self.valid_epoch, self.train_epoch]:
            e.metrics[1].__name__ = "IoU_Class0"
            e.metrics[2].__name__ = "IoU_Class1"
            e.metrics[3].__name__ = "IoU_Class2"
            # e.metrics[4].__name__ = "Fscore_all_classes"  # error while setting
            # e.metrics[5].__name__ = "Fscore"

        self.max_score = 0
        self.train_logs_vec = []
        self.valid_logs_vec = []
        self.best_model = None

    def train(self, train_loader, valid_loader, device, cpu_device, lrs):
        number_of_epochs = len(lrs)

        for i in range(0, number_of_epochs):
            self.optimizer.param_groups[0]['lr'] = lrs[i]
            print(f"Learning rate {self.optimizer.param_groups[0]['lr']}")

            print(f'\nEpoch: {i}')
            self.train_logs_vec.append(self.train_epoch.run(train_loader))
            valid_logs = self.valid_epoch.run(valid_loader)
            self.valid_logs_vec.append(valid_logs)

            if self.max_score < valid_logs['iou_score']:
                self.max_score = valid_logs['iou_score']
                self.model.to(cpu_device)
                self.best_model = copy.deepcopy(self.model)
                self.model.to(device)

        #     if i % 5 == 0 or i == (number_of_epochs - 1):
        #         best_model.to(CPU_DEVICE)
        #         torch.save(model.state_dict(), f'/media/data/local/corn/processed_stride768_v2/segfromer/model_cpu_segformer_best_model')
        #         res = {'valid_logs_vec': valid_logs_vec, 'train_logs_vec': train_logs_vec}
        #         with open(f"/media/data/local/corn/processed_stride768_v2/segfromer/model_cpu_segformer_epoch{i}.json", 'w') as f:
        #             json.dump(res, f)
        #         best_model.to(DEVICE)

        self.res.set(
            lrs=lrs,
            train_logs_vec=self.train_logs_vec,
            valid_logs_vec=self.valid_logs_vec)

    def run_test(self, test_loader, device):
        model = self.best_model
        model.to(device)

        test_log = self.test_epoch.run(test_loader)
        print(f'test_log = {test_log}')
        self.res.set(test_log=test_log)


def manual_prediction_test(test_loader, model, device, model_params):
    number_of_batches = len(test_loader)

    healthy_field_ground_truth_pix = 0
    damage_ground_truth_pix = 0

    healthy_field_predicted_pix = 0
    damage_field_predicted_pix = 0

    damage_prediction_true_positives_pix = 0

    healthy_intersection_pix = 0
    healthy_union_pix = 0

    damage_intersection_pix = 0
    damage_union_pix = 0

    for i, (img_batch, mask_batch) in enumerate(test_loader):
        print(f'Batch {i} / {number_of_batches}')
        with torch.no_grad():
            model_output = model(img_batch.to(device)).to(device)
            if model_params.metrics_activation:
                activation = nn.Softmax2d()
                model_output = activation(model_output)

        for i in range(model_output.shape[0]):
            ground_truth_healthy_field = mask_batch[i, 0, :, :].numpy().astype(int)
            ground_truth_damage = mask_batch[i, 1, :, :].numpy().astype(int)

            predicted_healty_field = model_output[i, 0, :, :].numpy()
            predicted_damage = model_output[i, 1, :, :].numpy()
            predicted_healty_field = np.where(predicted_healty_field > 0.5, 1, 0)
            predicted_damage = np.where(predicted_damage > 0.5, 1, 0)

            healthy_field_ground_truth_pix += np.count_nonzero(ground_truth_healthy_field)
            damage_ground_truth_pix += np.count_nonzero(ground_truth_damage)

            healthy_field_predicted_pix += np.count_nonzero(predicted_healty_field)
            damage_field_predicted_pix += np.count_nonzero(predicted_damage)

            common_damage = np.logical_and(ground_truth_damage, predicted_damage)
            damage_prediction_true_positives_pix += np.count_nonzero(common_damage)

            common_healthy = np.logical_and(ground_truth_healthy_field, predicted_healty_field)
            damage_intersection_pix += np.count_nonzero(common_damage)
            healthy_intersection_pix += np.count_nonzero(common_healthy)
            damage_union_pix += np.count_nonzero(np.logical_or(ground_truth_damage, predicted_damage))
            healthy_union_pix += np.count_nonzero(np.logical_or(ground_truth_healthy_field, predicted_healty_field))

    total_ground_truth_pix = healthy_field_ground_truth_pix + damage_ground_truth_pix
    total_predicted_pix = healthy_field_predicted_pix + damage_field_predicted_pix

    iou_damage = damage_intersection_pix / damage_union_pix
    iou_healthy = healthy_intersection_pix / healthy_union_pix

    print(f'healthy_field_ground_truth = {healthy_field_ground_truth_pix / total_ground_truth_pix * 100:.2f} %')
    print(f'damage_ground_truth = {damage_ground_truth_pix / total_ground_truth_pix * 100:.2f} %')

    print(f'healthy_field_predicted = {healthy_field_predicted_pix / total_predicted_pix * 100:.2f} %')
    print(f'damage_field_predicted = {damage_field_predicted_pix / total_predicted_pix * 100:.2f} %')

    print(
        f'damage_prediction_true_positives/damage_field_predicted = {damage_prediction_true_positives_pix / damage_field_predicted_pix * 100:.2f} %')

    print(f'iou_damage = {iou_damage:.3f}')
    print(f'iou_healthy = {iou_healthy:.3f}')
