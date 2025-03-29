"""
Class to load the CARRADA dataset
From: https://github.com/valeoai/MVRSS
"""
import json
import os
from utils.paths_collector import Paths

class Carrada:
    """Class to load CARRADA dataset"""

    def __init__(self, config_model):
        self.paths = Paths().get()
        if 'dataset_cfg' not in config_model:
            self.warehouse = self.paths['warehouse']
            self.carrada = self.paths['carrada']
        else:
            self.warehouse = config_model['dataset_cfg']['warehouse']
            self.carrada = config_model['dataset_cfg']['carrada']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self._split()

    def _load_data_seq_ref(self):
        path = os.path.join(self.carrada, 'data_seq_ref.json')
        with open(path, 'r') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = os.path.join(self.carrada, 'light_dataset_frame_oriented.json')
        with open(path, 'r') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        for sequence in self.annotations.keys():
            split = self.data_seq_ref[sequence]['split']
            if split == 'Train':
                self.train[sequence] = self.annotations[sequence]
            elif split == 'Validation':
                self.validation[sequence] = self.annotations[sequence]
            elif split == 'Test':
                self.test[sequence] = self.annotations[sequence]
            else:
                raise TypeError('Type {} is not supported for splits.'.format(split))

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))

