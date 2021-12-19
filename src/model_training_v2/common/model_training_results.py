import copy
import datetime
import json
import logging
from dataclasses import dataclass, field
import os
import pickle

import git
from matplotlib import pyplot as plt

from model_training_v2.common import model_definition


def default_to_json(o):
    try:
        return o.__dict__
    except:
        return repr(o)


@dataclass
class ModelTrainingResults:

    model_params: model_definition.ModelParams
    commit_hash: str = ''
    figures: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        repo = git.Repo(search_parent_directories=True)
        self.commit_hash = repo.head.object.hexsha
        print(f'Commit hash: {self.commit_hash}')

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, self.model_params.get_model_file_name() + '_result.json')
        data = copy.deepcopy(self.__dict__)
        data['model_params'].model_type = repr(data['model_params'].model_type)
        data['model_params'].loss_fnc = repr(data['model_params'].loss_fnc)
        data['figures'] = data['figures'].keys()
        data['date'] = datetime.datetime.now()

        with open(file_path, 'wt') as file:
            json.dump(data, file, default=default_to_json, sort_keys=True, indent=2)

        self._save_figures(dir_path)

    def _save_figures(self, dir_path):
        try:
            figures_dir = os.path.join(dir_path, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            for name, fig in self.figures.items():
                pickle_figure_path = os.path.join(figures_dir, name + '.pickle')
                with open(pickle_figure_path, 'wb') as file:
                    pickle.dump(fig, file)

                png_figure_path = os.path.join(figures_dir, name + '.png')
                fig.savefig(png_figure_path, dpi=300)
        except:
            logging.exception('Saving figures failed!')

    def set(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def add_fig(self, **kwargs):
        for key, val in kwargs.items():
            self.figures[key] = val


def load_pickled_figure(pickle_fig_path):
    with open(pickle_fig_path, 'rb') as file:
        figx = pickle.load(file)

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = figx
    figx.set_canvas(new_manager.canvas)
    # figx.show()

    # read the data from the plot
    # figx.axes[0].images[0].get_array().shape


if __name__ == '__main__':
    res = ModelTrainingResults(model_params=model_definition.ModelParams(model_definition.ModelType.UNET_PLUS_PLUS))
    res.set(a=3)
    res.save('/tmp/aaa')
