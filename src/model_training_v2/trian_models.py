import os
import sys

import cv2
import torch
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt


plt.ioff()


CPU_DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
DEVICE



sys.path.append('/home/przemek/Projects/pp/corn-field-damage/src')
TILES_BASE_DIR = '/media/data/local/corn/new/tiles_stride_768/'
BASE_OUTPUT_DIR = '/media/data/local/corn/out/122_01_22_tmp'


# sys.path.append('/home/przemek/Projects/corn-damage-segmentation/src')
# TILES_BASE_DIR = '/home/przemek/Projects/data/corn/tiles_stride_768'
# BASE_OUTPUT_DIR = '/home/przemek/Projects/data/corn/out/22_01_22'


TILE_SIZE = 512

import model_training_v2.common.dataset_preparation as dataset_preparation
import model_training_v2.common.corn_dataset as corn_dataset
import model_training_v2.common.model_definition as model_definition
import model_training_v2.common.model_training_results as model_training_results
import model_training_v2.common.model_training as model_training
import model_training_v2.common.plot_graphs as plot_graphs





for model_type in [
    model_definition.ModelType.UNET_PLUS_PLUS__RESNET18,
    model_definition.ModelType.UNET_PLUS_PLUS__RESNET34,
    model_definition.ModelType.UNET_PLUS_PLUS__RESNET50,
    model_definition.ModelType.UNET_PLUS_PLUS__RESNET101]:

    dataset_preparation.init_seeds(778)
    # model_type = model_definition.ModelType.UNET_PLUS_PLUS__EFFICIENT_NET_B3


    is_ndvi = 'ndvi' in TILES_BASE_DIR.lower()
    # is_ndvi = False

    image_channels = 4 if is_ndvi else 3


    model, model_params = model_definition.get_model_with_params(model_type, in_channels=image_channels, tile_size=TILE_SIZE)
    res = model_training_results.ModelTrainingResults(model_params=model_params)
    model_params.batch_size = 3


    print(f'model_params = {model_params}')
    print(model)


    train_loader, valid_loader, test_loader = corn_dataset.get_train_valid_test_loaders(
        dataset_name='dataset_split_demo.json',  # only a few samples for demo testing
        base_dir_path=TILES_BASE_DIR,
        batch_size=model_params.batch_size,
        mask_scalling=model_params.mask_scalling_factor,
        is_ndvi=is_ndvi)


    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, model_params.get_model_file_name())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    res.set(TILES_BASE_DIR=TILES_BASE_DIR)
    res.set(OUTPUT_DIR=OUTPUT_DIR)

    fig_example_imgages_testloader = plot_graphs.plot_images_from_dataloader(test_loader, seed=345, is_ndvi=is_ndvi, number_of_images=8)
    res.add_fig(fig_example_imgages_testloader=fig_example_imgages_testloader)

    print(res)


    model_trainer = model_training.ModelTrainer(
        model=model,
        device=DEVICE,
        model_params=model_params,
        res=res,
        )


    NUM = 1
    lrs = [0.0001] * NUM + [0.00003] * NUM + [0.00001] * NUM + [0.000003] * NUM + [0.000001] * NUM #  + [0.0000001] * NUM



    model_trainer.train(train_loader=train_loader, valid_loader=valid_loader, device=DEVICE, cpu_device=CPU_DEVICE, lrs=lrs)
    last_model = model


    figures = plot_graphs.plot_training_metrics(model_trainer=model_trainer, lrs=lrs)
    res.add_fig(**figures)


    model_trainer.run_test(test_loader=test_loader, device=DEVICE)


    model = model_trainer.best_model
    model_file_path = os.path.join(OUTPUT_DIR, 'model')
    model = model.to('cpu')
    torch.save(model.state_dict(), model_file_path)


    # model.load_state_dict(torch.load('/media/data/local/corn/processed_stride768/model_cpu'))
    # model.eval()
    # best_model = model


    figs = plot_graphs.plot_example_predictions(
        model=model,
        model_params=model_params,
        test_loader=test_loader,
        number_of_images=1,
        is_ndvi=is_ndvi)

    res.add_fig(**figs)


    res.save(dir_path=OUTPUT_DIR)
