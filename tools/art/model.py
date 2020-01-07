from abc import ABC, abstractmethod
import tensorflow as tf
from art.classifiers import TFClassifier
from utils.utils import load_checkpoint


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
        inputs = tf.compat.v1.placeholder(tf.float32,
                                          shape=[None, self._image_height, self._image_width, self._n_channels])
        target_ys = tf.compat.v1.placeholder(tf.float32, shape=[None, self._n_classes])

        logits = self.calculate_logits(inputs)

        target_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=target_ys, logits=logits))

        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, self._checkpoint_path)
        return TFClassifier(clip_values=(-1, 1), input_ph=inputs, output=logits, labels_ph=target_ys,
                            sess=session, loss=target_loss)

    @abstractmethod
    def calculate_logits(self, inputs):
        ...
