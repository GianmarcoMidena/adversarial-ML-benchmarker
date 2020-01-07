from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ready_models.tf_model import TFModel
from models.original_models import madry_model


class MadryModel(TFModel):
    def __init__(self, image_width=28, image_height=28, n_channels=1, n_classes=10, batch_size=32,
                 checkpoint_path="./ready_models/mnist_madry_ckpt"):
        super().__init__(image_width=image_width, image_height=image_height, n_channels=n_channels, n_classes=n_classes,
                         batch_size=batch_size, checkpoint_path=checkpoint_path)

    def calculate_logits(self, inputs):
        model = madry_model.MadryModel(n_classes=self.n_classes)
        output = model.fprop(inputs)
        return output['logits']
