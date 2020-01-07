from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception

from tools.cleverhans.model import Model


class InceptionV3Model(Model):
    def __init__(self, checkpoint_path, image_height, image_width, n_channels, n_classes):
        if n_classes == 1000:
            n_classes += 1
        super().__init__(image_height=image_height, image_width=image_width, n_channels=n_channels, n_classes=n_classes,
                         checkpoint_path=checkpoint_path)
        self._built = False

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self._built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(x_input, num_classes=self.n_classes,
                                                   is_training=False, reuse=reuse)
        self._built = True
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs

    def fprop(self, x, **kwargs):
        reuse = True if self._built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(x, num_classes=self.n_classes,
                                                        is_training=False, reuse=reuse)
        self._built = True
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        end_points['logits'] = logits
        end_points['probs'] = probs
        return end_points

    def get_layer_names(self):
        pass

    def make_input_placeholder(self):
        pass

    def make_label_placeholder(self):
        pass