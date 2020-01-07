from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception
from tools.art.model import Model


class InceptionV3Model(Model):
    def __init__(self, checkpoint_path, image_width=299, image_height=299, n_channels=3, n_classes=1001):
        if n_classes == 1000:
            n_classes += 1
        super().__init__(image_height=image_height, image_width=image_width, n_channels=n_channels,
                         n_classes=n_classes, checkpoint_path=checkpoint_path)

    def calculate_logits(self, inputs):
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, _ = inception.inception_v3(inputs, num_classes=self.n_classes, is_training=False)
        return logits
