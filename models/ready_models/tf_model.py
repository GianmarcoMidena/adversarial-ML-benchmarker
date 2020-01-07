from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from abc import abstractmethod
import os
from glob import glob

from models.ready_models.model import Model
from utils.utils import load_checkpoint


class TFModel(Model):
    def __init__(self, image_width, image_height, n_channels, n_classes, batch_size, checkpoint_path):
        super().__init__(image_width=image_width, image_height=image_height, n_channels=n_channels, n_classes=n_classes,
                         batch_size=batch_size)

        self._checkpoint_path = load_checkpoint(checkpoint_path)

        self._session = None

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        self._graph = tf.Graph()
        with self._graph.as_default():
            # Prepare graph
            self._x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, image_height, image_width, n_channels])

            logits = self.calculate_logits(self._x_input)

            self._predicted_labels = tf.argmax(logits, 1)

    @abstractmethod
    def calculate_logits(self, inputs):
        ...

    def predict(self, input_dir, logger):
        with tf.compat.v1.Session(graph=self._graph) as self._session:
            saver = tf.compat.v1.train.Saver()
            saver.restore(self._session, self._checkpoint_path)
            predictions = super().predict(input_dir, logger)
            return predictions

    def predict_batch(self, images):
        return self._session.run(self._predicted_labels, feed_dict={self._x_input: images})
