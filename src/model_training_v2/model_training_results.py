import copy
import datetime
import json
from dataclasses import dataclass
import os

import git

from model_definition import ModelParams, ModelType


def default_to_json(o):
    try:
        return o.__dict__
    except:
        return repr(o)


@dataclass
class ModelTrainingResults:

    model_params: ModelParams
    commit_hash: str = ''

    def __post_init__(self):
        repo = git.Repo(search_parent_directories=True)
        self.commit_hash = repo.head.object.hexsha
        print(f'Commit hash: {self.commit_hash}')

    def save(self, dir_path: str):
        file_path = os.path.join(dir_path, self.model_params.get_model_file_name() + '_result.json')
        data = copy.deepcopy(self.__dict__)
        data['model_params'].model_type = repr(data['model_params'].model_type)
        data['model_params'].loss_fnc = repr(data['model_params'].loss_fnc)
        data['date'] = datetime.datetime.now()

        with open(file_path, 'wt') as file:
            json.dump(data, file, default=default_to_json, sort_keys=True, indent=2)

    def set(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

if __name__ == '__main__':
    res = ModelTrainingResults(model_params=ModelParams(ModelType.UNET_PLUS_PLUS))
    res.set(a=3)
    res.save('/tmp/aaa')
