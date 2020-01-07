from abc import ABC
from cleverhans import model

from utils.utils import load_checkpoint


class Model(ABC, model.Model):
    def __init__(self, image_width, image_height, n_channels, n_classes, checkpoint_path):
        self._image_height = image_height
        self._image_width = image_width
        self._n_channels = n_channels
        self._n_classes = n_classes
        self._checkpoint_path = load_checkpoint(checkpoint_path)

    @property
    def image_height(self):
        return self._image_height

    @property
    def image_width(self):
        return self._image_width

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def checkpoint_path(self):
        return self._checkpoint_path
