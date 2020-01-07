from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception

from models.ready_models.tf_model import TFModel


class InceptionV3Model(TFModel):
    def __init__(self, image_width=299, image_height=299, n_channels=3, n_classes=1001, batch_size=32,
                 checkpoint_path="./ready_models/inception_v3_ckpt"):
        if n_classes == 1000:
            n_classes += 1
        super().__init__(image_width=image_width, image_height=image_height, n_channels=n_channels, n_classes=n_classes,
                         batch_size=batch_size, checkpoint_path=checkpoint_path)

    def calculate_logits(self, inputs):
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, _ = inception.inception_v3(self._x_input, num_classes=self.n_classes, is_training=False)
        return logits

    # def read(self, input_dir):
    #     return NormalizedImageIO.read(input_dir, batch_size=self._batch_size, image_height=self._image_height,
    #                                   image_width=self._image_width, n_channels=self._n_channels, recursive=True)
