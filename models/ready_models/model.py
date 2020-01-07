from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import pandas as pd
from glob import glob
from abc import ABC, abstractmethod

from utils.normalized_image_io import NormalizedImageIO


class Model(ABC):
    def __init__(self, image_width, image_height, n_channels, n_classes, batch_size):
        self._image_width = image_width
        self._image_height = image_height
        self._n_channels = n_channels
        self._n_classes = n_classes
        self._batch_size = batch_size

    def get_name(self):
        return self.__class__.__name__.replace("Model", "")

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

    def predict(self, input_dir, logger):
        predictions = pd.DataFrame()
        total_batches = math.ceil(len(glob(os.path.join(input_dir, "*.png"))) / self._batch_size)
        for i, (images, info) in enumerate(self.read(input_dir)):
            labels = self.predict_batch(images)
            info['label_pred'] = labels
            predictions = predictions.append(info, sort=False)
            logger.info("batch {} / {}".format(i+1, total_batches))
        if not predictions.empty:
            predictions['model'] = self.get_name()
            predictions['model'] = predictions['model'].astype(str)
            predictions['label_pred'] = predictions['label_pred'].astype(str)
        return predictions

    @abstractmethod
    def predict_batch(self, images):
        ...

    def read(self, input_dir):
        return NormalizedImageIO.read(input_dir, batch_size=self._batch_size, image_height=self._image_height,
                                      image_width=self._image_width, n_channels=self._n_channels)  # , recursive=True)
        # return ImageIO.read(input_dir, batch_size=self._batch_size, image_height=self._image_height,
        #                     image_width=self._image_width, n_channels=self._n_channels)
