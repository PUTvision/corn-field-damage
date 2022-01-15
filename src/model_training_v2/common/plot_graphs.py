import math
import random

from matplotlib import pyplot as plt
import torch
from torch import nn

from model_training_v2.common.model_definition import ModelParams
from model_training_v2.common.model_training import ModelTrainer


def plot_images_from_dataloader(data_loader, number_of_images=2, is_ndvi=False, seed=555):
    dataset = data_loader.dataset
    rng = random.Random(seed)

    columns = 4 + (1 if is_ndvi else 0)
    rows = number_of_images
    fig = plt.figure(figsize=(columns * 4, rows * 4))
    fig.suptitle('Example training dataset images')  # or plt.suptitle('Main title')

    for i in range(number_of_images):
        index = rng.randrange(len(dataset))
        image, mask = dataset[index]
        # image_path = dataset.img_path_at_index(index)  # TODO - add image_path to title

        if is_ndvi:
            image_rgb = image[0:3, :, :]
            image_ndvi = image[3, :, :]
        else:
            image_rgb = image

        column_terator = 0

        fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
        plt.imshow(image_rgb.transpose(1, 2, 0))
        plt.axis('off')
        plt.title('field RGB')
        column_terator += 1

        if is_ndvi:
            fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
            plt.imshow(image_ndvi)
            plt.axis('off')
            plt.title('field NDVI')
            column_terator += 1

        fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
        plt.imshow(mask[0, :, :])
        plt.axis('off')
        plt.title('healthy field mask')
        column_terator += 1

        fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
        plt.imshow(mask[1, :, :])
        plt.axis('off')
        plt.title('damaged field mask')
        column_terator += 1

        fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
        plt.imshow(mask[2, :, :])
        plt.axis('off')
        plt.title('out-of-field mask')
        column_terator += 1

    return fig


def plot_training_metrics(model_trainer: ModelTrainer, lrs):
    valid_logs_vec = model_trainer.valid_logs_vec
    train_logs_vec = model_trainer.train_logs_vec

    figures = {}
    for metric in valid_logs_vec[0].keys():
        train_metric_vec = [m[metric] for m in train_logs_vec]
        valid_metric_vec = [m[metric] for m in valid_logs_vec]

        fig = plt.figure()
        plt.plot(train_metric_vec)
        plt.plot(valid_metric_vec)

        plt.legend(['train', 'valid'])
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.grid()
        plt.show()

        figures['training_metric__' + metric] = fig

    fig = plt.figure()
    plt.grid()
    plt.plot(lrs)
    plt.yscale('log')
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    figures['learning_rates'] = fig

    return figures


def plot_example_predictions(model, model_params: ModelParams, test_loader, number_of_images=16, is_ndvi=False):
    vi = iter(test_loader)

    number_of_batches = int(math.ceil(number_of_images / model_params.batch_size))
    assert number_of_batches >= 1

    figs = {}

    for batch_number in range(number_of_batches):  # increase to get more images
        img_batch, mask_batch = next(vi)

        with torch.no_grad():
            #     model_output = model(img_batch.to(DEVICE))
            model_output = model(img_batch)
            if model_params.metrics_activation:
                assert model_params.metrics_activation == 'softmax2d'
                activation = nn.Softmax2d()
                model_output = activation(model_output)

        columns = 7 + (1 if is_ndvi else 0)
        rows = len(img_batch)
        fig = plt.figure(figsize=(columns * 4, rows * 4))

        for i in range(len(img_batch)):
            column_terator = 0

            if is_ndvi:
                image_rgb = img_batch[i].numpy()[0:3, :, :]
                image_ndvi = img_batch[i].numpy()[3, :, :]
            else:
                image_rgb = img_batch[i].numpy()

            fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
            plt.imshow(image_rgb.transpose([1, 2, 0]))
            plt.axis('off')
            plt.title('img')
            column_terator += 1

            if is_ndvi:
                fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
                plt.imshow(image_ndvi)
                plt.axis('off')
                plt.title('field NDVI')
                column_terator += 1

            fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
            plt.imshow(mask_batch[i][1].numpy())
            plt.axis('off')
            plt.title('original damage mask')
            column_terator += 1

            fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
            plt.imshow(model_output[i][1])
            plt.axis('off')
            plt.title('prediction damage')
            column_terator += 1

            fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
            cax = plt.imshow(model_output[i][1] - mask_batch[i][1], vmin=-1.1, vmax=1.1)
            plt.title('damage diff (predict-gt)')
            plt.axis('off')
            cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
            cbar.ax.set_yticklabels(['false negative', 'true', 'false positive'])
            column_terator += 1

            fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
            plt.imshow(model_output[i][0])
            plt.title('prediction healty field')
            plt.axis('off')
            column_terator += 1

            fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
            plt.imshow(model_output[i][1] > 0.5)
            plt.axis('off')
            plt.title('prediction damage \nthresholded (prob>0.5)')
            column_terator += 1

            fig.add_subplot(rows, columns, 1 + i * columns + column_terator)
            plt.imshow(model_output[i][1] > 0.3)
            plt.axis('off')
            plt.title('prediction damage \nthresholded (prob>0.3)')
            column_terator += 1

        figs[f'test_predictions_{batch_number}'] = fig
        plt.show()

    print('check wheher classification for one pixel makes sense...')
    one_pixel_classes = model_output[0][:, 33, 33]
    print(f'sum({one_pixel_classes} = {sum(one_pixel_classes)}')

    return figs
