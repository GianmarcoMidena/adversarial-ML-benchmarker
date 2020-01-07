from abc import ABC, abstractmethod
import tensorflow as tf
from utils.utils import load_checkpoint

from foolbox.models import TensorFlowModel


class Model(ABC):
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

    def __call__(self, session):
        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, self._image_height, self._image_width,
                                                             self._n_channels])

        logits = self.calculate_logits(inputs)

        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, self._checkpoint_path)
        return TensorFlowModel(inputs=inputs, logits=logits, bounds=(-1, 1))

    @abstractmethod
    def calculate_logits(self, inputs):
        ...
