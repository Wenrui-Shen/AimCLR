import random

import numpy as np
import pickle
import torch

from . import tools


class Feeder(torch.utils.data.Dataset):
    """Two-view feeder dedicated to ReSA/OSE pretraining.

    Each view independently walks the configured augmentation sequence.  Every
    augmentation is applied with the same Bernoulli probability, so multiple
    transforms may be composed in one view and every view gets a fresh draw.
    """

    _SUPPORTED_AUGMENTATIONS = (
        'temporal_crop',
        'shear',
        'rotation',
    )

    def __init__(self, data_path, label_path, shear_amplitude=0.5,
                 temperal_padding_ratio=6, mmap=True, return_index=False,
                 augmentation_methods=None, augmentation_probability=0.5):
        self.data_path = data_path
        self.label_path = label_path
        self.return_index = return_index
        self.shear_amplitude = float(shear_amplitude)
        self.temperal_padding_ratio = int(temperal_padding_ratio)
        if augmentation_methods is None:
            augmentation_methods = list(self._SUPPORTED_AUGMENTATIONS)
        self.augmentation_methods = tuple(augmentation_methods)
        unknown = [
            name for name in self.augmentation_methods
            if name not in self._SUPPORTED_AUGMENTATIONS
        ]
        if unknown:
            raise ValueError(
                'Unsupported ReSA/OSE augmentations: {}'.format(unknown))
        if len(set(self.augmentation_methods)) != len(
                self.augmentation_methods):
            raise ValueError('ReSA/OSE augmentations must not be repeated')
        self.augmentation_probability = float(augmentation_probability)
        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError('augmentation_probability must be in [0, 1]')
        self.load_data(mmap)

    def load_data(self, mmap):
        with open(self.label_path, 'rb') as file:
            self.sample_name, self.label = pickle.load(file)
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def _apply_augmentation(self, data_numpy, name):
        if name == 'temporal_crop':
            if self.temperal_padding_ratio > 0:
                return tools.temperal_crop(
                    data_numpy, self.temperal_padding_ratio)
            return data_numpy
        if name == 'shear':
            if self.shear_amplitude > 0:
                return tools.shear(data_numpy, self.shear_amplitude)
            return data_numpy
        if name == 'rotation':
            return tools.random_rotate(data_numpy)
        raise ValueError('Unsupported ReSA/OSE augmentation: {}'.format(name))

    def _aug(self, data_numpy):
        for name in self.augmentation_methods:
            if random.random() < self.augmentation_probability:
                data_numpy = self._apply_augmentation(data_numpy, name)
        return data_numpy

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        view_a = self._aug(data_numpy)
        view_b = self._aug(data_numpy)
        if self.return_index:
            return [view_a, view_b], label, index
        return [view_a, view_b], label
